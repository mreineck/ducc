/* Copyright (C) 2022 Max-Planck-Society
   Authors: Martin Reinecke, Philipp Arras */

class Baselines_GPU_prep
  {
  public:
    sycl::buffer<UVW,1> buf_uvw;
    sycl::buffer<double,1> buf_freq;

    Baselines_GPU_prep(const Baselines &bl)
      : buf_uvw(make_sycl_buffer(bl.getUVW_raw())),
        buf_freq(make_sycl_buffer(bl.get_f_over_c())) {}
  };

class Baselines_GPU
  {
  protected:
    sycl::accessor<UVW,1,sycl::access::mode::read> acc_uvw;
    sycl::accessor<double,1,sycl::access::mode::read> acc_f_over_c;

  public:
    Baselines_GPU(Baselines_GPU_prep &prep, sycl::handler &cgh)
      : acc_uvw(prep.buf_uvw.template get_access<sycl::access::mode::read>(cgh)),
        acc_f_over_c(prep.buf_freq.template get_access<sycl::access::mode::read>(cgh))
      {}

    UVW effectiveCoord(size_t row, size_t chan) const
      {
      double f = acc_f_over_c[chan];
      return acc_uvw[row]*f;
      }
    double absEffectiveW(size_t row, size_t chan) const
      { return sycl::fabs(acc_uvw[row].w*acc_f_over_c[chan]); }
    UVW baseCoord(size_t row) const
      { return acc_uvw[row]; }
    double ffact(size_t chan) const
      { return acc_f_over_c[chan];}
    size_t Nrows() const { return acc_uvw.get_range().get(0); }
    size_t Nchannels() const { return acc_f_over_c.get_range().get(0); }
  };

class IndexComputer0
  {
  public:
    struct Tileinfo { uint16_t tile_u, tile_v; };
    struct Blockinfo { uint32_t limits, startidx; };
    static constexpr size_t chunksize=1024;
    bool store_tiles;
    vector<uint32_t> row_gpu;
    vector<uint16_t> chbegin_gpu;
    vector<Tileinfo> tileinfo;
    vector<uint16_t> minplane_gpu;
    vector<uint32_t> vissum_gpu;
    vector<Blockinfo> blockinfo;

    IndexComputer0(const VVR &ranges, bool do_wgridding, bool store_tiles_)
      : store_tiles(store_tiles_)
      {
      size_t nranges=0;
      for (const auto &rng: ranges)
        nranges+=rng.second.size();
      row_gpu.reserve(nranges);
      chbegin_gpu.reserve(nranges);
      vissum_gpu.reserve(nranges+1);
      size_t isamp=0, curtile_u=~uint16_t(0), curtile_v=~uint16_t(0), curminplane=~uint16_t(0);
      size_t accum=0;

      // if necessary, resize some vectors to size 1, because SYCL is unhappy otherwise
      if (!do_wgridding)
        minplane_gpu.resize(1);
      if (!store_tiles)
        tileinfo.resize(1);
      for (const auto &rng: ranges)
        {
        if ((curtile_u!=rng.first.tile_u)||(curtile_v!=rng.first.tile_v)
          ||(curminplane!=rng.first.minplane))
          {
          blockinfo.push_back({uint32_t(row_gpu.size()), uint32_t(accum)});
          isamp=0;
          curtile_u = rng.first.tile_u;
          curtile_v = rng.first.tile_v;
          curminplane = rng.first.minplane;
          if (store_tiles)
            tileinfo.push_back({rng.first.tile_u, rng.first.tile_v});
          if (do_wgridding)
            minplane_gpu.push_back(rng.first.minplane);
          }
        for (const auto &rcr: rng.second)
          {
          auto nchan = size_t(rcr.ch_end-rcr.ch_begin);
          size_t curpos=0;
          while (curpos+chunksize-isamp<=nchan)
            {
            blockinfo.push_back({uint32_t(row_gpu.size()),
                                 uint32_t(blockinfo.back().startidx+chunksize)});
            if (store_tiles)
              tileinfo.push_back({rng.first.tile_u, rng.first.tile_v});
            if (do_wgridding)
              minplane_gpu.push_back(rng.first.minplane);
            curpos += chunksize-isamp;
            isamp = 0;
            }
          isamp += nchan-curpos;
          row_gpu.push_back(rcr.row);
          chbegin_gpu.push_back(rcr.ch_begin);
          vissum_gpu.push_back(accum);
          accum += nchan;
          }
        }
      blockinfo.push_back({uint32_t(row_gpu.size()), uint32_t(accum)});
      vissum_gpu.push_back(accum);
      }
  };
class IndexComputer: public IndexComputer0
  {
  public:
    sycl::buffer<uint32_t, 1> buf_row;
    sycl::buffer<uint16_t, 1> buf_chbegin;
    sycl::buffer<uint32_t, 1> buf_vissum;
    sycl::buffer<typename IndexComputer0::Blockinfo, 1> buf_blockinfo;
    sycl::buffer<typename IndexComputer0::Tileinfo, 1> buf_tileinfo;
    sycl::buffer<uint16_t, 1> buf_minplane;

    IndexComputer(const VVR &ranges, bool do_wgridding, bool store_tiles_)
      : IndexComputer0(ranges, do_wgridding, store_tiles_),
        buf_row(make_sycl_buffer(this->row_gpu)),
        buf_chbegin(make_sycl_buffer(this->chbegin_gpu)),
        buf_vissum(make_sycl_buffer(this->vissum_gpu)),
        buf_blockinfo(make_sycl_buffer(this->blockinfo)),
        buf_tileinfo(make_sycl_buffer(this->tileinfo)),
        buf_minplane(make_sycl_buffer(this->minplane_gpu))
        {}
  };

class GlobalCorrector
  {
  private:
    const Params &par;
    vector<double> cfu, cfv;
    sycl::buffer<double,1> bufcfu, bufcfv; 

    template<size_t maxsz> class Wcorrector
      {
      private:
        array<double, maxsz> x, wgtpsi;
        size_t n, supp;
    
      public:
        Wcorrector(const detail_gridding_kernel::KernelCorrection &corr)
          {
          const auto &x_ = corr.X();
          n = x_.size();
          MR_assert(n<=maxsz, "maxsz too small");
          const auto &wgtpsi_ = corr.Wgtpsi();
          supp = corr.Supp();
          for (size_t i=0; i<n; ++i)
            {
            x[i] = x_[i];
            wgtpsi[i] = wgtpsi_[i];
            }
          }
    
        double corfunc(double v) const
          {
          double tmp=0;
          for (size_t i=0; i<n; ++i)
            tmp += wgtpsi[i]*sycl::cos(pi*supp*v*x[i]);
          return 1./tmp;
          }
      };
   
    static double syclphase(double x, double y, double w, bool adjoint, double nshift)
      {
      double tmp = 1.-x-y;
      if (tmp<=0) return 0; // no phase factor beyond the horizon
      double nm1 = (-x-y)/(sycl::sqrt(tmp)+1); // more accurate form of sqrt(1-x-y)-1
      double phs = w*(nm1+nshift);
      if (adjoint) phs *= -1;
      if constexpr (is_same<Tcalc, double>::value)
        return twopi*phs;
      // we are reducing accuracy, so let's better do range reduction first
      return twopi*(phs-sycl::floor(phs));
      }
   
  public:
    GlobalCorrector(const Params &par_)
      : par(par_),
        cfu(par.krn->corfunc(par.nxdirty/2+1, 1./par.nu, par.nthreads)),
        cfv(par.krn->corfunc(par.nydirty/2+1, 1./par.nv, par.nthreads)),
        bufcfu(make_sycl_buffer(cfu)),
        bufcfv(make_sycl_buffer(cfv))
      {}

    void corr_degrid_narrow_field(sycl::queue &q,
      sycl::buffer<Tcalc, 2> &bufdirty, sycl::buffer<complex<Tcalc>, 2> &bufgrid)
      {
      // copy to grid and apply kernel correction
      q.submit([&](sycl::handler &cgh)
        {
        auto accdirty{bufdirty.template get_access<sycl::access::mode::read>(cgh)};
        auto acccfu{bufcfu.template get_access<sycl::access::mode::read>(cgh)};
        auto acccfv{bufcfv.template get_access<sycl::access::mode::read>(cgh)};
        auto accgrid{bufgrid.template get_access<sycl::access::mode::write>(cgh)};
        cgh.parallel_for(sycl::range<2>(par.nxdirty, par.nydirty), [accdirty,acccfu,acccfv,accgrid,nxdirty=par.nxdirty,nydirty=par.nydirty,nu=par.nu,nv=par.nv](sycl::item<2> item)
          {
          auto i = item.get_id(0);
          auto j = item.get_id(1);
          int icfu = sycl::abs(int(nxdirty/2)-int(i));
          int icfv = sycl::abs(int(nydirty/2)-int(j));
          size_t i2 = nu-nxdirty/2+i;
          if (i2>=nu) i2-=nu;
          size_t j2 = nv-nydirty/2+j;
          if (j2>=nv) j2-=nv;
          auto fctu = acccfu[icfu];
          auto fctv = acccfv[icfv];
          accgrid[i2][j2] = accdirty[i][j]*Tcalc(fctu*fctv);
          });
        });
      }

    void corr_grid_narrow_field(sycl::queue &q,
      sycl::buffer<complex<Tcalc>, 2> &bufgrid, sycl::buffer<Tcalc, 2> &bufdirty)
      {
      // copy to dirty image and apply kernel correction
      q.submit([&](sycl::handler &cgh)
        {
        auto accdirty{bufdirty.template get_access<sycl::access::mode::discard_write>(cgh)};
        auto acccfu{bufcfu.template get_access<sycl::access::mode::read>(cgh)};
        auto acccfv{bufcfv.template get_access<sycl::access::mode::read>(cgh)};
        auto accgrid{bufgrid.template get_access<sycl::access::mode::read>(cgh)};
        cgh.parallel_for(sycl::range<2>(par.nxdirty, par.nydirty), [accdirty,acccfu,acccfv,accgrid,nxdirty=par.nxdirty,nydirty=par.nydirty,nu=par.nu,nv=par.nv](sycl::item<2> item)
          {
          auto i = item.get_id(0);
          auto j = item.get_id(1);
          int icfu = sycl::abs(int(nxdirty/2)-int(i));
          int icfv = sycl::abs(int(nydirty/2)-int(j));
          size_t i2 = nu-nxdirty/2+i;
          if (i2>=nu) i2-=nu;
          size_t j2 = nv-nydirty/2+j;
          if (j2>=nv) j2-=nv;
          auto fctu = acccfu[icfu];
          auto fctv = acccfv[icfv];
          accdirty[i][j] = (accgrid[i2][j2]*Tcalc(fctu*fctv)).real();
          });
        });
      }

    void apply_global_corrections(sycl::queue &q,
      sycl::buffer<Tcalc, 2> &bufdirty)
      {
      // apply global corrections to dirty image on GPU
      q.submit([&](sycl::handler &cgh)
        {
        Wcorrector<30> wcorr(par.krn->Corr());
        auto accdirty{bufdirty.template get_access<sycl::access::mode::read_write>(cgh)};
        auto acccfu{bufcfu.template get_access<sycl::access::mode::read>(cgh)};
        auto acccfv{bufcfv.template get_access<sycl::access::mode::read>(cgh)};
        double x0 = par.lshift-0.5*par.nxdirty*par.pixsize_x,
               y0 = par.mshift-0.5*par.nydirty*par.pixsize_y;
        cgh.parallel_for(sycl::range<2>(par.nxdirty, par.nydirty), [nxdirty=par.nxdirty,nydirty=par.nydirty,accdirty,acccfu,acccfv,pixsize_x=par.pixsize_x,pixsize_y=par.pixsize_y,x0,y0,divide_by_n=par.divide_by_n,wcorr,nshift=par.nshift,dw=par.dw](sycl::item<2> item)
          {
          auto i = item.get_id(0);
          auto j = item.get_id(1);
          double fx = sqr(x0+i*pixsize_x);
          double fy = sqr(y0+j*pixsize_y);
          double fct;
          auto tmp = 1-fx-fy;
          if (tmp>=0)
            {
            auto nm1 = (-fx-fy)/(sycl::sqrt(tmp)+1); // accurate form of sqrt(1-x-y)-1
            fct = wcorr.corfunc((nm1+nshift)*dw);
            if (divide_by_n)
              fct /= nm1+1;
            }
          else // beyond the horizon, don't really know what to do here
            fct = divide_by_n ? 0 : wcorr.corfunc((sycl::sqrt(-tmp)-1)*dw);

          int icfu = sycl::abs(int(nxdirty/2)-int(i));
          int icfv = sycl::abs(int(nydirty/2)-int(j));
          accdirty[i][j]*=Tcalc(fct*acccfu[icfu]*acccfv[icfv]);
          });
        });
      }
    void degridding_wscreen(sycl::queue &q, double w,
      sycl::buffer<Tcalc, 2> &bufdirty, sycl::buffer<complex<Tcalc>, 2> &bufgrid)
      {
      // copy to grid and apply wscreen
      q.submit([&](sycl::handler &cgh)
        {
        auto accdirty{bufdirty.template get_access<sycl::access::mode::read>(cgh)};
        auto accgrid{bufgrid.template get_access<sycl::access::mode::write>(cgh)};
        double x0 = par.lshift-0.5*par.nxdirty*par.pixsize_x,
               y0 = par.mshift-0.5*par.nydirty*par.pixsize_y;
        cgh.parallel_for(sycl::range<2>(par.nxdirty, par.nydirty), [nxdirty=par.nxdirty, nydirty=par.nydirty, nu=par.nu, nv=par.nv, pixsize_x=par.pixsize_x, pixsize_y=par.pixsize_y,nshift=par.nshift,accgrid,accdirty,x0,y0,w](sycl::item<2> item)
          {
          auto i = item.get_id(0);
          auto j = item.get_id(1);
          size_t i2 = nu-nxdirty/2+i;
          if (i2>=nu) i2-=nu;
          size_t j2 = nv-nydirty/2+j;
          if (j2>=nv) j2-=nv;
          double fx = sqr(x0+i*pixsize_x);
          double fy = sqr(y0+j*pixsize_y);
          double myphase = syclphase(fx, fy, w, false, nshift);
          accgrid[i2][j2] = complex<Tcalc>(sycl::cos(myphase),sycl::sin(myphase))*accdirty[i][j];
          });
        });
      }
    void gridding_wscreen(sycl::queue &q, double w,
      sycl::buffer<complex<Tcalc>, 2> &bufgrid, sycl::buffer<Tcalc, 2> &bufdirty)
      {
      q.submit([&](sycl::handler &cgh)
        {
        auto accdirty{bufdirty.template get_access<sycl::access::mode::read_write>(cgh)};
        auto accgrid{bufgrid.template get_access<sycl::access::mode::read>(cgh)};
        double x0 = par.lshift-0.5*par.nxdirty*par.pixsize_x,
               y0 = par.mshift-0.5*par.nydirty*par.pixsize_y;
        cgh.parallel_for(sycl::range<2>(par.nxdirty, par.nydirty), [nxdirty=par.nxdirty, nydirty=par.nydirty, nu=par.nu, nv=par.nv, pixsize_x=par.pixsize_x, pixsize_y=par.pixsize_y,nshift=par.nshift,accgrid,accdirty,x0,y0,w](sycl::item<2> item)
          {
          auto i = item.get_id(0);
          auto j = item.get_id(1);
          size_t i2 = nu-nxdirty/2+i;
          if (i2>=nu) i2-=nu;
          size_t j2 = nv-nydirty/2+j;
          if (j2>=nv) j2-=nv;
          double fx = sqr(x0+i*pixsize_x);
          double fy = sqr(y0+j*pixsize_y);
          double myphase = syclphase(fx, fy, w, true, nshift);
          accdirty[i][j] += sycl::cos(myphase)*accgrid[i2][j2].real()
                           -sycl::sin(myphase)*accgrid[i2][j2].imag();
          });
        });
      }
  };
class RowchanComputer
  {
  protected:
    sycl::accessor<typename IndexComputer0::Blockinfo,1,sycl::access::mode::read> acc_blockinfo;
    sycl::accessor<uint32_t,1,sycl::access::mode::read> acc_vissum;
    sycl::accessor<uint32_t,1,sycl::access::mode::read> acc_row;
    sycl::accessor<uint16_t,1,sycl::access::mode::read> acc_chbegin;

  public:
    RowchanComputer(IndexComputer &idxcomp, sycl::handler &cgh)
      : acc_blockinfo(idxcomp.buf_blockinfo.template get_access<sycl::access::mode::read>(cgh)),
        acc_vissum(idxcomp.buf_vissum.template get_access<sycl::access::mode::read>(cgh)),
        acc_row(idxcomp.buf_row.template get_access<sycl::access::mode::read>(cgh)),
        acc_chbegin(idxcomp.buf_chbegin.template get_access<sycl::access::mode::read>(cgh))
      {}

    void getRowChan(size_t iblock, size_t iwork, size_t &irow, size_t &ichan) const
      {
      auto xlo = acc_blockinfo[iblock].limits;
      auto xhi = acc_blockinfo[iblock+1].limits;
      auto wanted = acc_blockinfo[iblock].startidx+iwork;
      if (wanted>=acc_blockinfo[iblock+1].startidx)
        { irow = ~size_t(0); return; }  // nothing to do for this item
      while (xlo+1<xhi)  // bisection search
        {
        auto xmid = (xlo+xhi)/2;
        (acc_vissum[xmid]<=wanted) ? xlo=xmid : xhi=xmid;
        }
      if (acc_vissum[xhi]<=wanted)
        xlo = xhi;
      irow = acc_row[xlo];
      ichan = acc_chbegin[xlo] + (wanted-acc_vissum[xlo]);
      }
  };

template<typename T> class KernelComputer
  {
  protected:
    sycl::accessor<T,1,sycl::access::mode::read> acc_coeff;
    size_t supp, D;

  public:
    KernelComputer(sycl::buffer<T,1> &buf_coeff, size_t supp_, sycl::handler &cgh)
      : acc_coeff(buf_coeff.template get_access<sycl::access::mode::read>(cgh)),
        supp(supp_), D(supp_+3) {}
    template<size_t Supp> inline void compute_uv(T ufrac, T vfrac, array<T,Supp> &ku, array<T,Supp> &kv) const
      {
//      if (Supp<supp) throw runtime_error("bad array size");
      auto x0 = T(ufrac)*T(-2)+T(supp-1);
      auto y0 = T(vfrac)*T(-2)+T(supp-1);
      for (size_t i=0; i<supp; ++i)
        {
        Tcalc resu=acc_coeff[i], resv=acc_coeff[i];
        for (size_t j=1; j<=D; ++j)
          {
          resu = resu*x0 + acc_coeff[j*supp+i];
          resv = resv*y0 + acc_coeff[j*supp+i];
          }
        ku[i] = resu;
        kv[i] = resv;
        }
      }
    template<size_t Supp> inline void compute_uvw(T ufrac, T vfrac, T wval, size_t nth, array<T,Supp> &ku, array<T,Supp> &kv) const
      {
//      if (Supp<supp) throw runtime_error("bad array size");
      auto x0 = T(ufrac)*T(-2)+T(supp-1);
      auto y0 = T(vfrac)*T(-2)+T(supp-1);
      auto z0 = T(wval-nth)*T(2)+T(supp-1);
      Tcalc resw=acc_coeff[nth];
      for (size_t j=1; j<=D; ++j)
        resw = resw*z0 + acc_coeff[j*supp+nth];
      for (size_t i=0; i<supp; ++i)
        {
        Tcalc resu=acc_coeff[i], resv=acc_coeff[i];
        for (size_t j=1; j<=D; ++j)
          {
          resu = resu*x0 + acc_coeff[j*supp+i];
          resv = resv*y0 + acc_coeff[j*supp+i];
          }
        ku[i] = resu*resw;
        kv[i] = resv;
        }
      }
  };

class CoordCalculator
  {
  private:
    size_t nu, nv;
    int maxiu0, maxiv0;
    double pixsize_x, pixsize_y, ushift, vshift;

  public:
    CoordCalculator (size_t nu_, size_t nv_, int maxiu0_, int maxiv0_, double pixsize_x_, double pixsize_y_, double ushift_, double vshift_)
      : nu(nu_), nv(nv_), maxiu0(maxiu0_), maxiv0(maxiv0_), pixsize_x(pixsize_x_), pixsize_y(pixsize_y_), ushift(ushift_), vshift(vshift_) {}

    [[gnu::always_inline]] void getpix(double u_in, double v_in, double &u, double &v, int &iu0, int &iv0) const
      {
      u = u_in*pixsize_x;
      u = (u-sycl::floor(u))*nu;
      iu0 = std::min(int(u+ushift)-int(nu), maxiu0);
      u -= iu0;
      v = v_in*pixsize_y;
      v = (v-sycl::floor(v))*nv;
      iv0 = std::min(int(v+vshift)-int(nv), maxiv0);
      v -= iv0;
      }
  };

    void dirty2x_gpu()
      {
      timers.push("GPU degridding");
      if (do_wgridding)
        {
#if (defined(DUCC0_HAVE_SYCL))
        { // Device buffer scope
        sycl::queue q{sycl::default_selector()};

        auto bufdirty(make_sycl_buffer(dirty_in));
        // grid (only on GPU)
        sycl::buffer<complex<Tcalc>, 2> bufgrid{sycl::range<2>(nu,nv)};

        Baselines_GPU_prep bl_prep(bl);

        auto bufvis(make_sycl_buffer(ms_out));
        const auto &dcoef(krn->Coeff());
        vector<Tcalc> coef(dcoef.size());
        for (size_t i=0;i<coef.size(); ++i) coef[i] = Tcalc(dcoef[i]);
        auto bufcoef(make_sycl_buffer(coef));

        sycl_zero_buffer(q, bufvis);

        // build index structure
        IndexComputer idxcomp(ranges, do_wgridding, false);
        // apply global corrections to dirty image on GPU
        GlobalCorrector globcorr(*this);
        globcorr.apply_global_corrections(q, bufdirty);

        CoordCalculator ccalc(nu, nv, maxiu0, maxiv0, pixsize_x, pixsize_y, ushift,vshift);

        for (size_t pl=0; pl<nplanes; ++pl)
          {
          double w = wmin+pl*dw;
          vector<size_t> blidx;
          for (size_t i=0; i<idxcomp.minplane_gpu.size(); ++i)
            {
            auto minpl = idxcomp.minplane_gpu[i];
            if ((pl>=minpl) && (pl<minpl+supp))
              blidx.push_back(i);
            }
          auto bufblidx(make_sycl_buffer(blidx));
          sycl_zero_buffer(q, bufgrid);

          globcorr.degridding_wscreen(q, w, bufdirty, bufgrid);

          // FFT
          sycl_c2c(q, bufgrid, true);

          constexpr size_t blksz = 32768;
          for (size_t ofs=0; ofs<blidx.size(); ofs+=blksz)
            {
            q.submit([&](sycl::handler &cgh)
              {
              Baselines_GPU blloc(bl_prep, cgh);
              KernelComputer<Tcalc> kcomp(bufcoef, supp, cgh);
              RowchanComputer rccomp(idxcomp,cgh);

              auto acc_minplane{idxcomp.buf_minplane.template get_access<sycl::access::mode::read>(cgh)};
              auto accblidx{bufblidx.template get_access<sycl::access::mode::read>(cgh)};
              auto accgrid{bufgrid.template get_access<sycl::access::mode::read>(cgh)};
              auto accvis{bufvis.template get_access<sycl::access::mode::write>(cgh)};

              constexpr size_t n_workitems = 32;
              sycl::range<2> global(min(blksz,blidx.size()-ofs), n_workitems);
              sycl::range<2> local(1, n_workitems);
              cgh.parallel_for(sycl::nd_range(global, local), [accgrid,accvis,nu=nu,nv=nv,supp=supp,shifting=shifting,lshift=lshift,mshift=mshift,nshift=nshift,rccomp,blloc,ccalc,kcomp,pl,acc_minplane,w,dw=dw,ofs,accblidx](sycl::nd_item<2> item)
                {
                auto iblock = accblidx[item.get_global_id(0)+ofs];
                auto minplane = acc_minplane[iblock];

                for (auto iwork=item.get_global_id(1); ; iwork+=item.get_global_range(1))
                  {
                  size_t irow, ichan;
                  rccomp.getRowChan(iblock, iwork, irow, ichan);
                  if (irow==~size_t(0)) return;
      
                  auto coord = blloc.effectiveCoord(irow, ichan);
                  auto imflip = coord.FixW();
      
                  // compute fractional and integer indices in "grid"
                  double ufrac,vfrac;
                  int iu0, iv0;
                  ccalc.getpix(coord.u, coord.v, ufrac, vfrac, iu0, iv0);
      
                  // compute kernel values
                  array<Tcalc, 16> ukrn, vkrn;
                  size_t nth=pl-minplane;
                  auto wval=Tcalc((w-coord.w)/dw);
                  kcomp.compute_uvw(ufrac, vfrac, wval, nth, ukrn, vkrn);
      
                  // loop over supp*supp pixels from "grid"
                  complex<Tcalc> res=0;
                  auto iustart=size_t((iu0+nu)%nu);
                  auto ivstart=size_t((iv0+nv)%nv);
                  for (size_t i=0, realiu=iustart; i<supp;
                       ++i, realiu = (realiu+1<nu) ? realiu+1 : 0)
                    {
                    complex<Tcalc> tmp = 0;
                    for (size_t j=0, realiv=ivstart; j<supp;
                         ++j, realiv = (realiv+1<nv) ? realiv+1 : 0)
                      tmp += vkrn[j]*accgrid[realiu][realiv];
                    res += ukrn[i]*tmp;
                    }
                  res.imag(res.imag()*imflip);
      
                  if (shifting)
                    {
                    // apply phase
                    double fct = coord.u*lshift + coord.v*mshift + coord.w*nshift;
                    if constexpr (is_same<double, Tcalc>::value)
                      fct*=twopi;
                    else
                      // we are reducing accuracy,
                      // so let's better do range reduction first
                      fct = twopi*(fct-sycl::floor(fct));
                    complex<Tcalc> phase(sycl::cos(Tcalc(fct)), -imflip*sycl::sin(Tcalc(fct)));
                    res *= phase;
                    }
                  accvis[irow][ichan] += res;
                  }
                });
              });
            }
          q.wait();
          } // end of loop over planes
        }  // end of device buffer scope, buffers are written back

        if (wgt.stride(0)!=0)  // we need to apply weights!
          execParallel(bl.Nrows(), nthreads, [&](size_t lo, size_t hi)
            {
            auto nchan = bl.Nchannels();
            for (auto irow=lo; irow<hi; ++irow)
              for (size_t ichan=0; ichan<nchan; ++ichan)
                ms_out(irow, ichan) *= wgt(irow, ichan);
            });
#else
        MR_fail("CUDA not found");
#endif
        }
      else
        {
#if (defined(DUCC0_HAVE_SYCL))
        { // Device buffer scope
        sycl::queue q{sycl::default_selector()};

        auto bufdirty(make_sycl_buffer(dirty_in));
        // grid (only on GPU)
        sycl::buffer<complex<Tcalc>, 2> bufgrid{sycl::range<2>(nu,nv)};

        Baselines_GPU_prep bl_prep(bl);
        auto bufvis(make_sycl_buffer(ms_out));
        const auto &dcoef(krn->Coeff());
        vector<Tcalc> coef(dcoef.size());
        for (size_t i=0;i<coef.size(); ++i) coef[i] = Tcalc(dcoef[i]);
        auto bufcoef(make_sycl_buffer(coef));

        sycl_zero_buffer(q, bufvis);
        sycl_zero_buffer(q, bufgrid);

        {
        GlobalCorrector globcorr(*this);
        globcorr.corr_degrid_narrow_field(q, bufdirty, bufgrid);
        }
        // FFT
        sycl_c2c(q, bufgrid, true);

        // build index structure
        IndexComputer idxcomp(ranges, do_wgridding, false);
        CoordCalculator ccalc(nu, nv, maxiu0, maxiv0, pixsize_x, pixsize_y, ushift,vshift);

        constexpr size_t blksz = 32768;
        size_t nblock = idxcomp.blockinfo.size()-1;
        for (size_t ofs=0; ofs<nblock; ofs+= blksz)
          {
          q.submit([&](sycl::handler &cgh)
            {
            Baselines_GPU blloc(bl_prep, cgh);
            KernelComputer<Tcalc> kcomp(bufcoef, supp, cgh);
            RowchanComputer rccomp(idxcomp, cgh);
  
            auto accgrid{bufgrid.template get_access<sycl::access::mode::read>(cgh)};
            auto accvis{bufvis.template get_access<sycl::access::mode::write>(cgh)};
  
            constexpr size_t n_workitems = 512;
            sycl::range<2> global(min(blksz,nblock-ofs), n_workitems);
            sycl::range<2> local(1, n_workitems);
            cgh.parallel_for(sycl::nd_range<2>(global, local), [accgrid,accvis,nu=nu,nv=nv,supp=supp,shifting=shifting,lshift=lshift,mshift=mshift,rccomp,blloc,ccalc,kcomp,ofs](sycl::nd_item<2> item)
              {
              auto iblock = item.get_global_id(0)+ofs;
  
              for (auto iwork=item.get_global_id(1); ; iwork+=item.get_global_range(1))
                {
                size_t irow, ichan;
                rccomp.getRowChan(iblock, iwork, irow, ichan);
                if (irow==~size_t(0)) return;
    
                auto coord = blloc.effectiveCoord(irow, ichan);
                auto imflip = coord.FixW();
    
                // compute fractional and integer indices in "grid"
                double ufrac,vfrac;
                int iu0, iv0;
                ccalc.getpix(coord.u, coord.v, ufrac, vfrac, iu0, iv0);
    
                // compute kernel values
                array<Tcalc, 16> ukrn, vkrn;
                kcomp.compute_uv(ufrac, vfrac, ukrn, vkrn);
    
                // loop over supp*supp pixels from "grid"
                complex<Tcalc> res=0;
                auto iustart=size_t((iu0+nu)%nu);
                auto ivstart=size_t((iv0+nv)%nv);
                for (size_t i=0, realiu=iustart; i<supp;
                     ++i, realiu = (realiu+1<nu)?realiu+1 : 0)
                  {
                  complex<Tcalc> tmp = 0;
                  for (size_t j=0, realiv=ivstart; j<supp;
                       ++j, realiv = (realiv+1<nv)?realiv+1 : 0)
                    tmp += vkrn[j]*accgrid[realiu][realiv];
                  res += ukrn[i]*tmp;
                  }
                res.imag(res.imag()*imflip);
    
                if (shifting)
                  {
                  // apply phase
                  double fct = coord.u*lshift + coord.v*mshift;
                  if constexpr (is_same<double, Tcalc>::value)
                    fct*=twopi;
                  else
                    // we are reducing accuracy,
                    // so let's better do range reduction first
                    fct = twopi*(fct-sycl::floor(fct));
                  complex<Tcalc> phase(sycl::cos(Tcalc(fct)), -imflip*sycl::sin(Tcalc(fct)));
                  res *= phase;
                  }
                accvis[irow][ichan] = res;
                }
              });
            });
          }
        }  // end of device buffer scope, buffers are written back
        if (wgt.stride(0)!=0)  // we need to apply weights!
          execParallel(bl.Nrows(), nthreads, [&](size_t lo, size_t hi)
            {
            auto nchan = bl.Nchannels();
            for (auto irow=lo; irow<hi; ++irow)
              for (size_t ichan=0; ichan<nchan; ++ichan)
                ms_out(irow, ichan) *= wgt(irow, ichan);
            });
#else
        MR_fail("CUDA not found");
#endif
        }
      timers.pop();
      }

    void x2dirty_gpu()
      {
      timers.push("GPU gridding");
      if (do_wgridding)
#if (defined(DUCC0_HAVE_SYCL))
        {
        bool do_weights = (wgt.stride(0)!=0);
        { // Device buffer scope
        sycl::queue q{sycl::default_selector()};

        auto bufdirty(make_sycl_buffer(dirty_out));
        sycl_zero_buffer(q, bufdirty);

        // grid (only on GPU)
        sycl::buffer<complex<Tcalc>, 2> bufgrid{sycl::range<2>(nu,nv)};
        sycl::buffer<Tcalc, 3> bufgridr{bufgrid.template reinterpret<Tcalc,3>(sycl::range<3>(nu,nv,2))};

        Baselines_GPU_prep bl_prep(bl);
        auto bufvis(make_sycl_buffer(ms_in));
        vmav<Tms,2> wgtx({1,1});
        auto bufwgt(make_sycl_buffer(do_weights ? wgt : wgtx));

        const auto &dcoef(krn->Coeff());
        vector<Tcalc> coef(dcoef.size());
        for (size_t i=0;i<coef.size(); ++i) coef[i] = Tcalc(dcoef[i]);
        auto bufcoef(make_sycl_buffer(coef));

        // build index structure
        IndexComputer idxcomp(ranges, do_wgridding, true);
        CoordCalculator ccalc(nu, nv, maxiu0, maxiv0, pixsize_x, pixsize_y, ushift,vshift);
        GlobalCorrector globcorr(*this);

        for (size_t pl=0; pl<nplanes; ++pl)
          {
          double w = wmin+pl*dw;
          vector<size_t> blidx;
          for (size_t i=0; i<idxcomp.minplane_gpu.size(); ++i)
            {
            auto minpl = idxcomp.minplane_gpu[i];
            if ((pl>=minpl) && (pl<minpl+supp))
              blidx.push_back(i);
            }
          auto bufblidx(make_sycl_buffer(blidx));

          sycl_zero_buffer(q, bufgrid);
          constexpr size_t blksz = 32768;
          for (size_t ofs=0; ofs<blidx.size(); ofs+= blksz)
            {
            q.submit([&](sycl::handler &cgh)
              {
              Baselines_GPU blloc(bl_prep, cgh);
              KernelComputer<Tcalc> kcomp(bufcoef, supp, cgh);
              RowchanComputer rccomp(idxcomp,cgh);
  
              auto acc_tileinfo{idxcomp.buf_tileinfo.template get_access<sycl::access::mode::read>(cgh)};
              auto acc_minplane{idxcomp.buf_minplane.template get_access<sycl::access::mode::read>(cgh)};
              auto accblidx{bufblidx.template get_access<sycl::access::mode::read>(cgh)};
              auto accgridr{bufgridr.template get_access<sycl::access::mode::read_write>(cgh)};
              auto accvis{bufvis.template get_access<sycl::access::mode::read>(cgh)};
              auto accwgt{bufwgt.template get_access<sycl::access::mode::read>(cgh)};
  
              constexpr size_t n_workitems = 32;
              sycl::range<2> global(min(blksz,blidx.size()-ofs), n_workitems);
              sycl::range<2> local(1, n_workitems);
              int nsafe = (supp+1)/2;
              size_t sidelen = 2*nsafe+(1<<logsquare);
              my_local_accessor<Tcalc,3> tile({sidelen,sidelen,2}, cgh);

              cgh.parallel_for<class grid_w>(sycl::nd_range(global,local), [accgridr,accvis,nu=nu,nv=nv,supp=supp,shifting=shifting,lshift=lshift,mshift=mshift,nshift=nshift,rccomp,blloc,ccalc,kcomp,pl,acc_minplane,sidelen,nsafe,acc_tileinfo,tile,w,dw=dw,ofs,accblidx,accwgt,do_weights](sycl::nd_item<2> item)
                {
                auto iblock = accblidx[item.get_global_id(0)+ofs];
                auto minplane = acc_minplane[iblock];
  
                // preparation
                for (size_t i=item.get_global_id(1); i<sidelen*sidelen; i+=item.get_local_range(1))
                  {
                  size_t iu = i/sidelen, iv = i%sidelen;
                  tile[iu][iv][0]=Tcalc(0);
                  tile[iu][iv][1]=Tcalc(0);
                  }
                item.barrier(sycl::access::fence_space::local_space);
  
                for (auto iwork=item.get_global_id(1); ; iwork+=item.get_global_range(1))
                  {
                  size_t irow, ichan;
                  rccomp.getRowChan(iblock, iwork, irow, ichan);
                  if (irow==~size_t(0)) break;  // work done 
  
                  auto coord = blloc.effectiveCoord(irow, ichan);
                  auto imflip = coord.FixW();
    
                  // compute fractional and integer indices in "grid"
                  double ufrac,vfrac;
                  int iu0, iv0;
                  ccalc.getpix(coord.u, coord.v, ufrac, vfrac, iu0, iv0);
    
                  // compute kernel values
                  array<Tcalc, 16> ukrn, vkrn;
                  size_t nth=pl-minplane;
                  auto wval=Tcalc((w-coord.w)/dw);
                  kcomp.compute_uvw(ufrac, vfrac, wval, nth, ukrn, vkrn);
    
                  // loop over supp*supp pixels from "grid"
                  complex<Tcalc> val=accvis[irow][ichan];
                  if (do_weights) val *= accwgt[irow][ichan];
                  if (shifting)
                    {
                    // apply phase
                    double fct = coord.u*lshift + coord.v*mshift + coord.w*nshift;
                    if constexpr (is_same<double, Tcalc>::value)
                      fct*=twopi;
                    else
                      // we are reducing accuracy,
                      // so let's better do range reduction first
                      fct = twopi*(fct-sycl::floor(fct));
                    complex<Tcalc> phase(sycl::cos(Tcalc(fct)), imflip*sycl::sin(Tcalc(fct)));
                    val *= phase;
                    }
                  val.imag(val.imag()*imflip);
  
                  int bu0=((((iu0+nsafe)>>logsquare)<<logsquare))-nsafe;
                  int bv0=((((iv0+nsafe)>>logsquare)<<logsquare))-nsafe;
                  for (size_t i=0, ipos=iu0-bu0; i<supp; ++i, ++ipos)
                    {
                    auto tmp = ukrn[i]*val;
                    for (size_t j=0, jpos=iv0-bv0; j<supp; ++j, ++jpos)
                      {
                      auto tmp2 = vkrn[j]*tmp;
                      my_atomic_ref_l<Tcalc> rr(tile[ipos][jpos][0]);
                      rr.fetch_add(tmp2.real());
                      my_atomic_ref_l<Tcalc> ri(tile[ipos][jpos][1]);
                      ri.fetch_add(tmp2.imag());
                      }
                    }
                  }
  
                // add local buffer back to global buffer
                auto u_tile = acc_tileinfo[iblock].tile_u;
                auto v_tile = acc_tileinfo[iblock].tile_v;
                item.barrier(sycl::access::fence_space::local_space);
                for (size_t i=item.get_global_id(1); i<sidelen*sidelen; i+=item.get_global_range(1))
                  {
                  size_t iu = i/sidelen, iv = i%sidelen;
                  size_t igu = (iu+u_tile*(1<<logsquare)+nu-nsafe)%nu;
                  size_t igv = (iv+v_tile*(1<<logsquare)+nv-nsafe)%nv;
  
                  my_atomic_ref<Tcalc> rr(accgridr[igu][igv][0]);
                  rr.fetch_add(tile[iu][iv][0]);
                  my_atomic_ref<Tcalc> ri(accgridr[igu][igv][1]);
                  ri.fetch_add(tile[iu][iv][1]);
                  }
                });
              });
            }
          // FFT
          sycl_c2c(q, bufgrid, false);

          globcorr.gridding_wscreen(q, w, bufgrid, bufdirty);
          q.wait();
          } // end of loop over planes

        // apply global corrections to dirty image on GPU
        globcorr.apply_global_corrections(q, bufdirty);
        }  // end of device buffer scope, buffers are written back
        }
#else
        MR_fail("CUDA not found");
#endif
      else
        {
#if (defined(DUCC0_HAVE_SYCL))
        bool do_weights = (wgt.stride(0)!=0);

        { // Device buffer scope
        sycl::queue q{sycl::default_selector()};
        // dirty image
        auto bufdirty(make_sycl_buffer(dirty_out));
        // grid (only on GPU)
        sycl::buffer<complex<Tcalc>, 2> bufgrid{sycl::range<2>(nu,nv)};
        sycl::buffer<Tcalc, 3> bufgridr{bufgrid.template reinterpret<Tcalc,3>(sycl::range<3>(nu,nv,2))};

        Baselines_GPU_prep bl_prep(bl);
        auto bufvis(make_sycl_buffer(ms_in));
        vmav<Tms,2> wgtx({1,1});
        auto bufwgt(make_sycl_buffer(do_weights ? wgt : wgtx));

        const auto &dcoef(krn->Coeff());
        vector<Tcalc> coef(dcoef.size());
        for (size_t i=0;i<coef.size(); ++i) coef[i] = Tcalc(dcoef[i]);
        auto bufcoef(make_sycl_buffer(coef));

        sycl_zero_buffer(q, bufgrid);

        // build index structure
        IndexComputer idxcomp(ranges, do_wgridding, true);
        CoordCalculator ccalc(nu, nv, maxiu0, maxiv0, pixsize_x, pixsize_y, ushift,vshift);

        constexpr size_t blksz = 32768;
        size_t nblock = idxcomp.blockinfo.size()-1;
        for (size_t ofs=0; ofs<nblock; ofs+= blksz)
          {
          q.submit([&](sycl::handler &cgh)
            {
            Baselines_GPU blloc(bl_prep, cgh);
            KernelComputer<Tcalc> kcomp(bufcoef, supp, cgh);
            RowchanComputer rccomp(idxcomp, cgh);
  
            auto acc_tileinfo{idxcomp.buf_tileinfo.template get_access<sycl::access::mode::read>(cgh)};
  
            auto accgridr{bufgridr.template get_access<sycl::access::mode::read_write>(cgh)};
            auto accvis{bufvis.template get_access<sycl::access::mode::read>(cgh)};
            auto accwgt{bufwgt.template get_access<sycl::access::mode::read>(cgh)};
  
            constexpr size_t n_workitems = 512;
            sycl::range<2> global(min(blksz,nblock-ofs), n_workitems);
            sycl::range<2> local(1, n_workitems);
            int nsafe = (supp+1)/2;
            size_t sidelen = 2*nsafe+(1<<logsquare);
            my_local_accessor<Tcalc,3> tile({sidelen,sidelen,2}, cgh);

            cgh.parallel_for(sycl::nd_range(global,local), [accgridr,accvis,acc_tileinfo,tile,nu=nu,nv=nv,supp=supp,shifting=shifting,lshift=lshift,mshift=mshift,rccomp,blloc,ccalc,kcomp,nsafe,sidelen,ofs,accwgt,do_weights](sycl::nd_item<2> item)
              {
              auto iblock = item.get_global_id(0)+ofs;
  
              // preparation: zero local buffer
              for (size_t i=item.get_global_id(1); i<sidelen*sidelen; i+=item.get_global_range(1))
                {
                size_t iu = i/sidelen, iv = i%sidelen;
                tile[iu][iv][0] = Tcalc(0);
                tile[iu][iv][1] = Tcalc(0);
                }
              item.barrier();
  
              for (auto iwork=item.get_global_id(1); ; iwork+=item.get_global_range(1))
                {
                size_t irow, ichan;
                rccomp.getRowChan(iblock, iwork, irow, ichan);
                if (irow==~size_t(0)) break;  // work done
  
                auto coord = blloc.effectiveCoord(irow, ichan);
                auto imflip = coord.FixW();
  
                // compute fractional and integer indices in "grid"
                double ufrac,vfrac;
                int iu0, iv0;
                ccalc.getpix(coord.u, coord.v, ufrac, vfrac, iu0, iv0);
  
                // compute kernel values
                array<Tcalc, 16> ukrn, vkrn;
                kcomp.compute_uv(ufrac, vfrac, ukrn, vkrn);
  
                // loop over supp*supp pixels from "grid"
                complex<Tcalc> val=accvis[irow][ichan];
                if(do_weights) val *= accwgt[irow][ichan];
                if (shifting)
                  {
                  // apply phase
                  double fct = coord.u*lshift + coord.v*mshift;
                  if constexpr (is_same<double, Tcalc>::value)
                    fct*=twopi;
                  else
                    // we are reducing accuracy,
                    // so let's better do range reduction first
                    fct = twopi*(fct-sycl::floor(fct));
                  complex<Tcalc> phase(sycl::cos(Tcalc(fct)), imflip*sycl::sin(Tcalc(fct)));
                  val *= phase;
                  }
                val.imag(val.imag()*imflip);
  
                int bu0=((((iu0+nsafe)>>logsquare)<<logsquare))-nsafe;
                int bv0=((((iv0+nsafe)>>logsquare)<<logsquare))-nsafe;
                for (size_t i=0, ipos=iu0-bu0; i<supp; ++i, ++ipos)
                  {
                  auto tmp = ukrn[i]*val;
                  for (size_t j=0, jpos=iv0-bv0; j<supp; ++j, ++jpos)
                    {
                    auto tmp2 = vkrn[j]*tmp;
                    my_atomic_ref_l<Tcalc> rr(tile[ipos][jpos][0]);
                    rr.fetch_add(tmp2.real());
                    my_atomic_ref_l<Tcalc> ri(tile[ipos][jpos][1]);
                    ri.fetch_add(tmp2.imag());
                    }
                  }
                }
  
              // add local buffer back to global buffer
              auto u_tile = acc_tileinfo[iblock].tile_u;
              auto v_tile = acc_tileinfo[iblock].tile_v;
              item.barrier();
              //size_t ofs = (supp-1)/2;
              for (size_t i=item.get_global_id(1); i<sidelen*sidelen; i+=item.get_global_range(1))
                {
                size_t iu = i/sidelen, iv = i%sidelen;
                size_t igu = (iu+u_tile*(1<<logsquare)+nu-nsafe)%nu;
                size_t igv = (iv+v_tile*(1<<logsquare)+nv-nsafe)%nv;
                my_atomic_ref<Tcalc> rr(accgridr[igu][igv][0]);
                rr.fetch_add(tile[iu][iv][0]);
                my_atomic_ref<Tcalc> ri(accgridr[igu][igv][1]);
                ri.fetch_add(tile[iu][iv][1]);
                }
              });
            });
          }

        // FFT
        sycl_c2c(q, bufgrid, false);

        {
        GlobalCorrector globcorr(*this);
        globcorr.corr_grid_narrow_field(q, bufgrid, bufdirty);
        }
        }  // end of device buffer scope, buffers are written back
#else
        MR_fail("CUDA not found");
#endif
        }
      timers.pop();
      }
