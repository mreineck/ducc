/* Copyright (C) 2022 Max-Planck-Society
   Authors: Martin Reinecke, Philipp Arras */

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
        tmp += wgtpsi[i]*cos(pi*supp*v*x[i]);
      return 1./tmp;
      }
  };

class Baselines_GPU_prep
  {
  public:
    sycl::buffer<double,2> buf_uvw;
    sycl::buffer<double,1> buf_freq;

    Baselines_GPU_prep(const Baselines &bl)
      : buf_uvw(reinterpret_cast<const double *>(bl.getUVW_raw().data()),
          sycl::range<2>(bl.Nrows(), 3),
          {sycl::property::buffer::use_host_ptr()}),
        buf_freq(make_sycl_buffer(bl.get_f_over_c())) {}
  };

class Baselines_GPU
  {
  protected:
    sycl::accessor<double,2,sycl::access::mode::read> acc_uvw;
    sycl::accessor<double,1,sycl::access::mode::read> acc_f_over_c;
    size_t nrows, nchan;

  public:
    Baselines_GPU(Baselines_GPU_prep &prep, sycl::handler &cgh)
      : acc_uvw(prep.buf_uvw.template get_access<sycl::access::mode::read>(cgh)),
        acc_f_over_c(prep.buf_freq.template get_access<sycl::access::mode::read>(cgh)),
        nrows(acc_uvw.get_range().get(0)),
        nchan(acc_f_over_c.get_range().get(0))
      {
      MR_assert(acc_uvw.get_range().get(1)==3, "dimension mismatch");
      }

    UVW effectiveCoord(size_t row, size_t chan) const
      {
      double f = acc_f_over_c[chan];
      return UVW(acc_uvw[row][0]*f,
                 acc_uvw[row][1]*f,
                 acc_uvw[row][2]*f);
      }
    double absEffectiveW(size_t row, size_t chan) const
      { return abs(acc_uvw[row][2]*acc_f_over_c[chan]); }
    UVW baseCoord(size_t row) const
      {
      return UVW(acc_uvw[row][0],
                 acc_uvw[row][1],
                 acc_uvw[row][2]);
      }
    double ffact(size_t chan) const
      { return acc_f_over_c[chan];}
    size_t Nrows() const { return nrows; }
    size_t Nchannels() const { return nchan; }
  };

class IndexComputer0
  {
  public:
    static constexpr size_t chunksize=1024;
    bool store_tiles;
    vector<uint32_t> row_gpu;
    vector<uint16_t> chbegin_gpu;
    vector<uint16_t> tile_u_gpu, tile_v_gpu;
    vector<uint16_t> minplane_gpu;
    vector<uint32_t> vissum_gpu;
    vector<uint32_t> blocklimits;
    vector<uint32_t> blockstartidx;

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
        {
        tile_u_gpu.resize(1);
        tile_v_gpu.resize(1);
        }
      for (const auto &rng: ranges)
        {
        if ((curtile_u!=rng.first.tile_u)||(curtile_v!=rng.first.tile_v)
          ||(curminplane!=rng.first.minplane))
          {
          blocklimits.push_back(row_gpu.size());
          blockstartidx.push_back(accum);
          isamp=0;
          curtile_u = rng.first.tile_u;
          curtile_v = rng.first.tile_v;
          curminplane = rng.first.minplane;
          if (store_tiles)
            {
            tile_u_gpu.push_back(rng.first.tile_u);
            tile_v_gpu.push_back(rng.first.tile_v);
            }
          if (do_wgridding)
            minplane_gpu.push_back(rng.first.minplane);
          }
        for (const auto &rcr: rng.second)
          {
          auto nchan = size_t(rcr.ch_end-rcr.ch_begin);
          size_t curpos=0;
          while (curpos+chunksize-isamp<=nchan)
            {
            blocklimits.push_back(row_gpu.size());
            blockstartidx.push_back(blockstartidx.back()+chunksize);
            if (store_tiles)
              {
              tile_u_gpu.push_back(rng.first.tile_u);
              tile_v_gpu.push_back(rng.first.tile_v);
              }
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
      blocklimits.push_back(row_gpu.size());
      blockstartidx.push_back(accum);
      vissum_gpu.push_back(accum);
      }
  };
class IndexComputer: public IndexComputer0
  {
  public:
    sycl::buffer<uint32_t, 1> buf_row;
    sycl::buffer<uint16_t, 1> buf_chbegin;
    sycl::buffer<uint32_t, 1> buf_vissum;
    sycl::buffer<uint32_t, 1> buf_blocklimits;
    sycl::buffer<uint32_t, 1> buf_blockstartidx;
    sycl::buffer<uint16_t, 1> buf_tileu;
    sycl::buffer<uint16_t, 1> buf_tilev;
    sycl::buffer<uint16_t, 1> buf_minplane;

    IndexComputer(const VVR &ranges, bool do_wgridding, bool store_tiles_)
      : IndexComputer0(ranges, do_wgridding, store_tiles_),
        buf_row(make_sycl_buffer(this->row_gpu)),
        buf_chbegin(make_sycl_buffer(this->chbegin_gpu)),
        buf_vissum(make_sycl_buffer(this->vissum_gpu)),
        buf_blocklimits(make_sycl_buffer(this->blocklimits)),
        buf_blockstartidx(make_sycl_buffer(this->blockstartidx)),
        buf_tileu(make_sycl_buffer(this->tile_u_gpu)),
        buf_tilev(make_sycl_buffer(this->tile_v_gpu)),
        buf_minplane(make_sycl_buffer(this->minplane_gpu))
        {}
  };

class RowchanComputer
  {
  protected:
    sycl::accessor<uint32_t,1,sycl::access::mode::read> acc_blocklimits;
    sycl::accessor<uint32_t,1,sycl::access::mode::read> acc_blockstartidx;
    sycl::accessor<uint32_t,1,sycl::access::mode::read> acc_vissum;
    sycl::accessor<uint32_t,1,sycl::access::mode::read> acc_row;
    sycl::accessor<uint16_t,1,sycl::access::mode::read> acc_chbegin;

  public:
    RowchanComputer(IndexComputer &idxcomp, sycl::handler &cgh)
      : acc_blocklimits(idxcomp.buf_blocklimits.template get_access<sycl::access::mode::read>(cgh)),
        acc_blockstartidx(idxcomp.buf_blockstartidx.template get_access<sycl::access::mode::read>(cgh)),
        acc_vissum(idxcomp.buf_vissum.template get_access<sycl::access::mode::read>(cgh)),
        acc_row(idxcomp.buf_row.template get_access<sycl::access::mode::read>(cgh)),
        acc_chbegin(idxcomp.buf_chbegin.template get_access<sycl::access::mode::read>(cgh))
      {}

    void getRowChan(size_t iblock, size_t iwork, size_t &irow, size_t &ichan) const
      {
      auto xlo = acc_blocklimits[iblock];
      auto xhi = acc_blocklimits[iblock+1];
      auto wanted = acc_blockstartidx[iblock]+iwork;
      if (wanted>=acc_blockstartidx[iblock+1])
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
    inline void compute_uv(T ufrac, T vfrac, array<T,16> &ku, array<T,16> &kv) const
      {
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
    inline void compute_uvw(T ufrac, T vfrac, T wval, size_t nth, array<T,16> &ku, array<T,16> &kv) const
      {
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
      u = (u-floor(u))*nu;
      iu0 = std::min(int(u+ushift)-int(nu), maxiu0);
      u -= iu0;
      v = v_in*pixsize_y;
      v = (v-floor(v))*nv;
      iv0 = std::min(int(v+vshift)-int(nv), maxiv0);
      v -= iv0;
      }
  };

    void dirty2x_gpu()
      {
      if (do_wgridding)
        {
#if (defined(DUCC0_HAVE_SYCL))
timers.push("GPU degridding");
          
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

        auto cfu = krn->corfunc(nxdirty/2+1, 1./nu, nthreads);
        auto cfv = krn->corfunc(nydirty/2+1, 1./nv, nthreads);
  // FIXME: cast to Timg
        auto bufcfu(make_sycl_buffer(cfu));
        auto bufcfv(make_sycl_buffer(cfv));

        // build index structure
        timers.push("index creation");
#ifdef BUFFERING
        IndexComputer idxcomp(ranges, do_wgridding, true);
#else
        IndexComputer idxcomp(ranges, do_wgridding, false);
#endif
        timers.pop();

        // applying global corrections to dirty image on GPU
        q.submit([&](sycl::handler &cgh)
          {
Wcorrector<30> wcorr(krn->Corr());
          auto accdirty{bufdirty.template get_access<sycl::access::mode::read_write>(cgh)};
          auto acccfu{bufcfu.template get_access<sycl::access::mode::read>(cgh)};
          auto acccfv{bufcfv.template get_access<sycl::access::mode::read>(cgh)};
          double x0 = lshift-0.5*nxdirty*pixsize_x,
                 y0 = mshift-0.5*nydirty*pixsize_y;
          cgh.parallel_for(sycl::range<2>(nxdirty, nydirty), [nxdirty=nxdirty,nydirty=nydirty,accdirty,acccfu,acccfv,pixsize_x=pixsize_x,pixsize_y=pixsize_y,x0,y0,divide_by_n=divide_by_n,wcorr,nshift=nshift,dw=dw](sycl::item<2> item)
            {
            auto i = item.get_id(0);
            auto j = item.get_id(1);
            double fx = sqr(x0+i*pixsize_x);
            double fy = sqr(y0+j*pixsize_y);
            double fct;
            auto tmp = 1-fx-fy;
            if (tmp>=0)
              {
              auto nm1 = (-fx-fy)/(sqrt(tmp)+1); // accurate form of sqrt(1-x-y)-1
              fct = wcorr.corfunc((nm1+nshift)*dw);
              if (divide_by_n)
                fct /= nm1+1;
              }
            else // beyond the horizon, don't really know what to do here
              fct = divide_by_n ? 0 : wcorr.corfunc((sqrt(-tmp)-1)*dw);

            int icfu = abs(int(nxdirty/2)-int(i));
            int icfv = abs(int(nydirty/2)-int(j));
            accdirty[i][j]*=Tcalc(fct*acccfu[icfu]*acccfv[icfv]);
            });
          });

        for (size_t pl=0; pl<nplanes; ++pl)
          {
//cout << "plane: " << pl << endl;
          double w = wmin+pl*dw;

          sycl_zero_buffer(q, bufgrid);

          // copying to grid and applying wscreen
          q.submit([&](sycl::handler &cgh)
            {
            auto accdirty{bufdirty.template get_access<sycl::access::mode::read>(cgh)};
            auto accgrid{bufgrid.template get_access<sycl::access::mode::write>(cgh)};
            double x0 = lshift-0.5*nxdirty*pixsize_x,
                   y0 = mshift-0.5*nydirty*pixsize_y;
            cgh.parallel_for(sycl::range<2>(nxdirty, nydirty), [nxdirty=nxdirty, nydirty=nydirty, nu=nu, nv=nv, pixsize_x=pixsize_x, pixsize_y=pixsize_y,nshift=nshift,accgrid,accdirty,x0,y0,w](sycl::item<2> item)
              {
              auto i = item.get_id(0);
              auto j = item.get_id(1);
              size_t i2 = nu-nxdirty/2+i;
              if (i2>=nu) i2-=nu;
              size_t j2 = nv-nydirty/2+j;
              if (j2>=nv) j2-=nv;
              double fx = sqr(x0+i*pixsize_x);
              double fy = sqr(y0+j*pixsize_y);
              double myphase = phase(fx, fy, w, false, nshift);
              accgrid[i2][j2] = complex<Tcalc>(polar(1., myphase))*accdirty[i][j];
              });
            });

          // FFT
          sycl_c2c(q, bufgrid, true);

#ifdef BUFFERING
          constexpr size_t blksz = 1024;
          for (size_t blockofs=0; blockofs<idxcomp.blocklimits.size()-1; blockofs+=blksz)
            {
            size_t blockend = min(blockofs+blksz,idxcomp.blocklimits.size()-1);
#endif
          q.submit([&](sycl::handler &cgh)
            {
            Baselines_GPU blloc(bl_prep, cgh);
            KernelComputer<Tcalc> kcomp(bufcoef, supp, cgh);
            CoordCalculator ccalc(nu, nv, maxiu0, maxiv0, pixsize_x, pixsize_y, ushift,vshift);
            RowchanComputer rccomp(idxcomp,cgh);
  
#ifdef BUFFERING
            auto acc_tileu{idxcomp.buf_tileu.template get_access<sycl::access::mode::read>(cgh)};
            auto acc_tilev{idxcomp.buf_tilev.template get_access<sycl::access::mode::read>(cgh)};
#endif
            auto acc_minplane{idxcomp.buf_minplane.template get_access<sycl::access::mode::read>(cgh)};
            auto accgrid{bufgrid.template get_access<sycl::access::mode::read>(cgh)};
            auto accvis{bufvis.template get_access<sycl::access::mode::write>(cgh)};
#ifdef BUFFERING
            sycl::range<2> global(blockend-blockofs, idxcomp.chunksize);
            sycl::range<2> local(1, idxcomp.chunksize);
            int nsafe = (supp+1)/2;
            size_t sidelen = 2*nsafe+(1<<logsquare);
#ifndef __INTEL_LLVM_COMPILER
            sycl::local_accessor<complex<Tcalc>,2> tile({sidelen,sidelen}, cgh);
#else
            sycl::accessor<complex<Tcalc>,2,sycl::access::mode::read_write, sycl::access::target::local> tile({sidelen,sidelen}, cgh);
#endif
            cgh.parallel_for(sycl::nd_range(global,local), [accgrid,accvis,nu=nu,nv=nv,supp=supp,shifting=shifting,lshift=lshift,mshift=mshift,nshift=nshift,rccomp,blloc,ccalc,kcomp,pl,acc_minplane,blockofs,sidelen,nsafe,acc_tileu,acc_tilev,tile,w,dw=dw](sycl::nd_item<2> item)
#else
            cgh.parallel_for(sycl::range<2>(idxcomp.blocklimits.size()-1, idxcomp.chunksize), [accgrid,accvis,nu=nu,nv=nv,supp=supp,shifting=shifting,lshift=lshift,mshift=mshift,nshift=nshift,rccomp,blloc,ccalc,kcomp,pl,acc_minplane,w,dw=dw](sycl::item<2> item)
#endif
              {
#ifdef BUFFERING
              auto iblock = item.get_global_id(0)+blockofs;
              auto iwork = item.get_local_id(1);
              auto minplane = acc_minplane[iblock];
              if ((pl<minplane) || (pl>=minplane+supp))  // plane not in range
                return;
              // preparation
              auto u_tile = acc_tileu[iblock];
              auto v_tile = acc_tilev[iblock];

              //size_t ofs = (supp-1)/2;
              for (size_t i=iwork; i<sidelen*sidelen; i+=item.get_local_range(1))
                {
                size_t iu = i/sidelen, iv = i%sidelen;
                tile[iu][iv] = accgrid[(iu+u_tile*(1<<logsquare)+nu-nsafe)%nu][(iv+v_tile*(1<<logsquare)+nv-nsafe)%nv];
                }
              item.barrier();
#else
              auto iblock = item.get_id(0);
              auto iwork = item.get_id(1);
              auto minplane = acc_minplane[iblock];
              if ((pl<minplane) || (pl>=minplane+supp))  // plane not in range
                return;
#endif
  
              size_t irow, ichan;
              rccomp.getRowChan(iblock, iwork, irow, ichan);
              if (irow==~size_t(0)) return;
  
              auto coord = blloc.effectiveCoord(irow, ichan);
              auto imflip = coord.FixW();
  
              // compute fractional and integer indices in "grid"
              double ufrac,vfrac;
              int iu0, iv0;
              ccalc.getpix( coord.u, coord.v, ufrac, vfrac, iu0, iv0);
  
              // compute kernel values
              array<Tcalc, 16> ukrn, vkrn;
size_t nth=pl-minplane;
auto wval=Tcalc((coord.w-w)/dw);
              kcomp.compute_uvw(ufrac, vfrac, wval, nth, ukrn, vkrn);
  
              // loop over supp*supp pixels from "grid"
              complex<Tcalc> res=0;
#ifdef BUFFERING
              int bu0=((((iu0+nsafe)>>logsquare)<<logsquare))-nsafe;
              int bv0=((((iv0+nsafe)>>logsquare)<<logsquare))-nsafe;
              for (size_t i=0; i<supp; ++i)
                {
                complex<Tcalc> tmp = 0;
                for (size_t j=0; j<supp; ++j)
                  tmp += vkrn[j]*tile[iu0-bu0+i][iv0-bv0+j];
                res += ukrn[i]*tmp;
                }
#else
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
#endif
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
                  fct = twopi*(fct-floor(fct));
                complex<Tcalc> phase(cos(Tcalc(fct)), -imflip*sin(Tcalc(fct)));
                res *= phase;
                }
              accvis[irow][ichan] += res;
              });
            });
#ifdef BUFFERING
}
#endif
          } // end of loop over planes
        }  // end of device buffer scope, buffers are written back
        timers.poppush("weight application");
        if (wgt.stride(0)!=0)  // we need to apply weights!
          execParallel(bl.Nrows(), nthreads, [&](size_t lo, size_t hi)
            {
            auto nchan = bl.Nchannels();
            for (auto irow=lo; irow<hi; ++irow)
              for (size_t ichan=0; ichan<nchan; ++ichan)
                ms_out(irow, ichan) *= wgt(irow, ichan);
            });
        timers.pop();
#else
        MR_fail("CUDA not found");
#endif
        }
      else
        {
#if (defined(DUCC0_HAVE_SYCL))
timers.push("GPU degridding");
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
        auto cfu = krn->corfunc(nxdirty/2+1, 1./nu, nthreads);
        auto cfv = krn->corfunc(nydirty/2+1, 1./nv, nthreads);
// FIXME: cast to Timg
        auto bufcfu(make_sycl_buffer(cfu));
        auto bufcfv(make_sycl_buffer(cfv));
        // copying to grid and applying correction
        q.submit([&](sycl::handler &cgh)
          {
          auto accdirty{bufdirty.template get_access<sycl::access::mode::read>(cgh)};
          auto acccfu{bufcfu.template get_access<sycl::access::mode::read>(cgh)};
          auto acccfv{bufcfv.template get_access<sycl::access::mode::read>(cgh)};
          auto accgrid{bufgrid.template get_access<sycl::access::mode::write>(cgh)};
          cgh.parallel_for(sycl::range<2>(nxdirty, nydirty), [accdirty,acccfu,acccfv,accgrid,nxdirty=nxdirty,nydirty=nydirty,nu=nu,nv=nv](sycl::item<2> item)
            {
            auto i = item.get_id(0);
            auto j = item.get_id(1);
            int icfu = abs(int(nxdirty/2)-int(i));
            int icfv = abs(int(nydirty/2)-int(j));
            size_t i2 = nu-nxdirty/2+i;
            if (i2>=nu) i2-=nu;
            size_t j2 = nv-nydirty/2+j;
            if (j2>=nv) j2-=nv;
            auto fctu = acccfu[icfu];
            auto fctv = acccfv[icfv];
            accgrid[i2][j2] = accdirty[i][j]*Tcalc(fctu*fctv);
            });
          });

        // FFT
        sycl_c2c(q, bufgrid, true);

        // build index structure
        timers.push("index creation");
#ifdef BUFFERING
        IndexComputer idxcomp(ranges, do_wgridding, true);
#else
        IndexComputer idxcomp(ranges, do_wgridding, false);
#endif
        timers.pop();

#ifdef BUFFERING
        constexpr size_t blksz = 1024;
        for (size_t blockofs=0; blockofs<idxcomp.blocklimits.size()-1; blockofs+=blksz)
          {
          size_t blockend = min(blockofs+blksz,idxcomp.blocklimits.size()-1);
#endif
        q.submit([&](sycl::handler &cgh)
          {
          Baselines_GPU blloc(bl_prep, cgh);
          KernelComputer<Tcalc> kcomp(bufcoef, supp, cgh);
          CoordCalculator ccalc(nu, nv, maxiu0, maxiv0, pixsize_x, pixsize_y, ushift,vshift);
          RowchanComputer rccomp(idxcomp, cgh);

#ifdef BUFFERING
          auto acc_tileu{idxcomp.buf_tileu.template get_access<sycl::access::mode::read>(cgh)};
          auto acc_tilev{idxcomp.buf_tilev.template get_access<sycl::access::mode::read>(cgh)};
#endif
          auto accgrid{bufgrid.template get_access<sycl::access::mode::read>(cgh)};
          auto accvis{bufvis.template get_access<sycl::access::mode::write>(cgh)};
#ifdef BUFFERING
          sycl::range<2> global(blockend-blockofs, idxcomp.chunksize);
          sycl::range<2> local(1, idxcomp.chunksize);
          int nsafe = (supp+1)/2;
          size_t sidelen = 2*nsafe+(1<<logsquare);
#ifndef __INTEL_LLVM_COMPILER
          sycl::local_accessor<complex<Tcalc>,2> tile({sidelen,sidelen}, cgh);
#else
          sycl::accessor<complex<Tcalc>,2,sycl::access::mode::read_write, sycl::access::target::local> tile({sidelen,sidelen}, cgh);
#endif
          cgh.parallel_for(sycl::nd_range(global,local), [accgrid,accvis,acc_tileu,acc_tilev,tile,nu=nu,nv=nv,supp=supp,shifting=shifting,lshift=lshift,mshift=mshift,rccomp,blloc,ccalc,kcomp,blockofs,nsafe,sidelen](sycl::nd_item<2> item)
#else
          cgh.parallel_for(sycl::range<2>(idxcomp.blocklimits.size()-1, idxcomp.chunksize), [accgrid,accvis,nu=nu,nv=nv,supp=supp,shifting=shifting,lshift=lshift,mshift=mshift,rccomp,blloc,ccalc,kcomp](sycl::item<2> item)
#endif
            {
#ifdef BUFFERING
            auto iblock = item.get_global_id(0)+blockofs;
            auto iwork = item.get_local_id(1);
            // preparation
            auto u_tile = acc_tileu[iblock];
            auto v_tile = acc_tilev[iblock];
            //size_t ofs = (supp-1)/2;
            for (size_t i=iwork; i<sidelen*sidelen; i+=item.get_local_range(1))
              {
              size_t iu = i/sidelen, iv = i%sidelen;
              tile[iu][iv] = accgrid[(iu+u_tile*(1<<logsquare)+nu-nsafe)%nu][(iv+v_tile*(1<<logsquare)+nv-nsafe)%nv];
              }
            item.barrier();
#else
            auto iblock = item.get_id(0);
            auto iwork = item.get_id(1);
#endif

            size_t irow, ichan;
            rccomp.getRowChan(iblock, iwork, irow, ichan);
            if (irow==~size_t(0)) return;

            auto coord = blloc.effectiveCoord(irow, ichan);
            auto imflip = coord.FixW();

            // compute fractional and integer indices in "grid"
            double ufrac,vfrac;
            int iu0, iv0;
            ccalc.getpix( coord.u, coord.v, ufrac, vfrac, iu0, iv0);

            // compute kernel values
            array<Tcalc, 16> ukrn, vkrn;
            kcomp.compute_uv(ufrac, vfrac, ukrn, vkrn);

            // loop over supp*supp pixels from "grid"
            complex<Tcalc> res=0;
#ifdef BUFFERING
            int bu0=((((iu0+nsafe)>>logsquare)<<logsquare))-nsafe;
            int bv0=((((iv0+nsafe)>>logsquare)<<logsquare))-nsafe;
            for (size_t i=0; i<supp; ++i)
              {
              complex<Tcalc> tmp = 0;
              for (size_t j=0; j<supp; ++j)
                tmp += vkrn[j]*tile[iu0-bu0+i][iv0-bv0+j];
              res += ukrn[i]*tmp;
              }
#else
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
#endif
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
                fct = twopi*(fct-floor(fct));
              complex<Tcalc> phase(cos(Tcalc(fct)), -imflip*sin(Tcalc(fct)));
              res *= phase;
              }
            accvis[irow][ichan] = res;
            });
          });
#ifdef BUFFERING
}
#endif
        }  // end of device buffer scope, buffers are written back
        timers.poppush("weight application");
        if (wgt.stride(0)!=0)  // we need to apply weights!
          execParallel(bl.Nrows(), nthreads, [&](size_t lo, size_t hi)
            {
            auto nchan = bl.Nchannels();
            for (auto irow=lo; irow<hi; ++irow)
              for (size_t ichan=0; ichan<nchan; ++ichan)
                ms_out(irow, ichan) *= wgt(irow, ichan);
            });
        timers.pop();
#else
        MR_fail("CUDA not found");
#endif
        }
      }



template<typename T> static void atomic_add(complex<T> &a, const complex<T> &b)
  {
  T *aptr = reinterpret_cast<T *>(&a);
#ifndef __INTEL_LLVM_COMPILER
  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> re(aptr[0]);
  re += b.real();
  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> im(aptr[1]);
  im += b.imag();
#else
  sycl::ext::oneapi::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,sycl::access::address_space::global_space> re(aptr[0]);
  re += b.real();
  sycl::ext::oneapi::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,sycl::access::address_space::global_space> im(aptr[1]);
  im += b.imag();
#endif
  }

    void x2dirty_gpu()
      {
      if (do_wgridding)
        MR_fail("not implemented");
      else
        {
#if (defined(DUCC0_HAVE_SYCL))
timers.push("GPU gridding");
        timers.push("weight application");
        bool do_weights = (wgt.stride(0)!=0);
        vmav<complex<Tms>,2> ms_tmp({do_weights ? bl.Nrows() : 1, do_weights ? bl.Nchannels() : 1});
        if (do_weights)
          mav_apply([](const complex<Tms> &a, const Tms &b, complex<Tms> &c)
            { c = a*b; }, nthreads, ms_in, wgt, ms_tmp);
        const cmav<complex<Tms>,2> &ms(do_weights ? ms_tmp : ms_in);
        timers.pop();

        { // Device buffer scope
        sycl::queue q{sycl::default_selector()};
        // dirty image
        auto bufdirty(make_sycl_buffer(dirty_out));
        // grid (only on GPU)
        sycl::buffer<complex<Tcalc>, 2> bufgrid{sycl::range<2>(nu,nv)};

        Baselines_GPU_prep bl_prep(bl);
        auto bufvis(make_sycl_buffer(ms));

        const auto &dcoef(krn->Coeff());
        vector<Tcalc> coef(dcoef.size());
        for (size_t i=0;i<coef.size(); ++i) coef[i] = Tcalc(dcoef[i]);
        auto bufcoef(make_sycl_buffer(coef));

        sycl_zero_buffer(q, bufgrid);
        auto cfu = krn->corfunc(nxdirty/2+1, 1./nu, nthreads);
        auto cfv = krn->corfunc(nydirty/2+1, 1./nv, nthreads);
// FIXME: cast to Timg
        auto bufcfu(make_sycl_buffer(cfu));
        auto bufcfv(make_sycl_buffer(cfv));

        // build index structure
        timers.push("index creation");
        IndexComputer idxcomp(ranges, do_wgridding, true);
        timers.pop();

        constexpr size_t blksz = 1024;
        for (size_t blockofs=0; blockofs<idxcomp.blocklimits.size()-1; blockofs+=blksz)
          {
          size_t blockend = min(blockofs+blksz,idxcomp.blocklimits.size()-1);
          q.submit([&](sycl::handler &cgh)
            {
            Baselines_GPU blloc(bl_prep, cgh);
            KernelComputer<Tcalc> kcomp(bufcoef, supp, cgh);
            CoordCalculator ccalc(nu, nv, maxiu0, maxiv0, pixsize_x, pixsize_y, ushift,vshift);
            RowchanComputer rccomp(idxcomp, cgh);

            auto acc_tileu{idxcomp.buf_tileu.template get_access<sycl::access::mode::read>(cgh)};
            auto acc_tilev{idxcomp.buf_tilev.template get_access<sycl::access::mode::read>(cgh)};

            auto accgrid{bufgrid.template get_access<sycl::access::mode::write>(cgh)};
            auto accvis{bufvis.template get_access<sycl::access::mode::read>(cgh)};

            sycl::range<2> global(blockend-blockofs, idxcomp.chunksize);
            sycl::range<2> local(1, idxcomp.chunksize);
            int nsafe = (supp+1)/2;
            size_t sidelen = 2*nsafe+(1<<logsquare);
#ifndef __INTEL_LLVM_COMPILER
            sycl::local_accessor<complex<Tcalc>,2> tile({sidelen,sidelen}, cgh);
#else
            sycl::accessor<complex<Tcalc>,2,sycl::access::mode::read_write, sycl::access::target::local> tile({sidelen,sidelen}, cgh);
#endif
            cgh.parallel_for(sycl::nd_range(global,local), [accgrid,accvis,acc_tileu,acc_tilev,tile,nu=nu,nv=nv,supp=supp,shifting=shifting,lshift=lshift,mshift=mshift,rccomp,blloc,ccalc,kcomp,blockofs,nsafe,sidelen](sycl::nd_item<2> item)
              {
              auto iblock = item.get_global_id(0)+blockofs;
              auto iwork = item.get_local_id(1);

              // preparation
              // zero local buffer (FIXME is this needed?)
              for (size_t i=iwork; i<sidelen*sidelen; i+=item.get_local_range(1))
                {
                size_t iu = i/sidelen, iv = i%sidelen;
                tile[iu][iv] = Tcalc(0);
                }
              item.barrier();

              size_t irow, ichan;
              rccomp.getRowChan(iblock, iwork, irow, ichan);
              if (irow!=~size_t(0))
                {
                auto coord = blloc.effectiveCoord(irow, ichan);
                auto imflip = coord.FixW();

                // compute fractional and integer indices in "grid"
                double ufrac,vfrac;
                int iu0, iv0;
                ccalc.getpix( coord.u, coord.v, ufrac, vfrac, iu0, iv0);

                // compute kernel values
                array<Tcalc, 16> ukrn, vkrn;
                kcomp.compute_uv(ufrac, vfrac, ukrn, vkrn);

                // loop over supp*supp pixels from "grid"
                complex<Tcalc> val=accvis[irow][ichan];
                if (shifting)
                  {
                  // apply phase
                  double fct = coord.u*lshift + coord.v*mshift;
                  if constexpr (is_same<double, Tcalc>::value)
                    fct*=twopi;
                  else
                    // we are reducing accuracy,
                    // so let's better do range reduction first
                    fct = twopi*(fct-floor(fct));
                  complex<Tcalc> phase(cos(Tcalc(fct)), imflip*sin(Tcalc(fct)));
                  val *= phase;
                  }
                val.imag(val.imag()*imflip);

                int bu0=((((iu0+nsafe)>>logsquare)<<logsquare))-nsafe;
                int bv0=((((iv0+nsafe)>>logsquare)<<logsquare))-nsafe;
                for (size_t i=0; i<supp; ++i)
                  {
                  auto tmp = ukrn[i]*val;
                  for (size_t j=0; j<supp; ++j)
                    atomic_add(tile[iu0-bu0+i][iv0-bv0+j], vkrn[j]*tmp);
                  }
                }

              // add local buffer back to global buffer
              auto u_tile = acc_tileu[iblock];
              auto v_tile = acc_tilev[iblock];
              item.barrier();
              //size_t ofs = (supp-1)/2;
              for (size_t i=iwork; i<sidelen*sidelen; i+=item.get_local_range(1))
                {
                size_t iu = i/sidelen, iv = i%sidelen;
                atomic_add(accgrid[(iu+u_tile*(1<<logsquare)+nu-nsafe)%nu][(iv+v_tile*(1<<logsquare)+nv-nsafe)%nv], tile[iu][iv]);
                }
              });
            });
          }
        // FFT
        sycl_c2c(q, bufgrid, false);  // FIXME normalization?

        // copying to dirty image and applying correction
        q.submit([&](sycl::handler &cgh)
          {
          auto accdirty{bufdirty.template get_access<sycl::access::mode::discard_write>(cgh)};
          auto acccfu{bufcfu.template get_access<sycl::access::mode::read>(cgh)};
          auto acccfv{bufcfv.template get_access<sycl::access::mode::read>(cgh)};
          auto accgrid{bufgrid.template get_access<sycl::access::mode::read>(cgh)};
          cgh.parallel_for(sycl::range<2>(nxdirty, nydirty), [accdirty,acccfu,acccfv,accgrid,nxdirty=nxdirty,nydirty=nydirty,nu=nu,nv=nv](sycl::item<2> item)
            {
            auto i = item.get_id(0);
            auto j = item.get_id(1);
            int icfu = abs(int(nxdirty/2)-int(i));
            int icfv = abs(int(nydirty/2)-int(j));
            size_t i2 = nu-nxdirty/2+i;
            if (i2>=nu) i2-=nu;
            size_t j2 = nv-nydirty/2+j;
            if (j2>=nv) j2-=nv;
            auto fctu = acccfu[icfu];
            auto fctv = acccfv[icfv];
            accdirty[i][j] = (accgrid[i2][j2]*Tcalc(fctu*fctv)).real();
            });
          });
        }  // end of device buffer scope, buffers are written back
        timers.pop();
#else
        MR_fail("CUDA not found");
#endif
        }
      }
