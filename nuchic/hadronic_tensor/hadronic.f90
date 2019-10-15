module hadronic_response
   implicit none
   integer*4, private, save :: xA,i_fg,np_del
   real*8, private, save :: xpf, xmn
   real*8, private, parameter :: pi=acos(-1.0d0),hbarc=197.327053d0,xmp=938.0d0,ppmax=1.0d0*1.e3
   real*8, private, parameter :: alpha=1.0d0/137.0d0
   real*8, private, allocatable :: pdel(:),pot_del(:)
contains

subroutine one_body(p1, pp1, q, ff, r_now)
    use dirac_matrices
    IMPLICIT NONE
    ! p1: momentum of incoming nucleon (E, px, py, pz)
    ! pp1: momentum of outgoing nucleon (E, px, py, pz)
    ! ff: form factors
    ! q: momentum transfer (E, px, py, pz)
    real*8, INTENT(IN) :: p1(4), pp1(4), q(4), ff(4)
    real*8, INTENT(OUT) :: r_now(5)
    real*8 :: rlp, rtp, rln, rtn, ff1p, ff1n, ff2p, ff2n

    print*, p1, pp1, q, ff

    call current_init_one_body(p1, pp1, q)
    call define_spinors()

    ff1p = 0.5d0*(ff(1)+ff(2))
    ff2p = 0.5d0*(ff(3)+ff(4))
    ff1n = 0.5d0*(-ff(1)+ff(2))
    ff2n = 0.5d0*(-ff(3)+ff(4))
    call det_Ja(ff1p, ff2p)
    call det_res1b(rlp, rtp)
    call det_Ja(ff1n, ff2n)
    call det_res1b(rln, rtn)

    r_now(1) = (rlp+rln)
    r_now(2) = (rtp+rtn)

end subroutine one_body

subroutine two_body(p1_4,p2_4,pp1_4,pp2_4,q_4,dp1,dp2,r_now)
    use dirac_matrices         
    use mathtool
    implicit none
    real*8, parameter :: lsq=0.71*1.e6,l3=3.5d0*1.e6,xma2=1.1025d0*1.e6
    real*8, parameter :: fstar=2.13d0,eps=20.0d0
    real*8 :: w
    real*8 :: q2,rho,norm,ca5,cv3,gep
    real*8 :: p1_4(4),p2_4(4),pp1_4(4),pp2_4(4),k2_4(4),k1_4(4),q_4(4),pp_4(4)
    real*8 :: p1,p2,pp1,ep1,ep2,arg
    real*8 :: k2e_4(4),k1e_4(4)
    real*8 :: r_cc_pi,r_cl_pi,r_ll_pi,r_t_pi,r_tp_pi
    real*8 :: r_cc_del,r_cl_del,r_ll_del,r_t_del,r_tp_del   
    real*8 :: r_cc_int,r_cl_int,r_ll_int,r_t_int,r_tp_int  
    real*8 :: dp1,dp2,delta_w
    real*8 :: tkin_pp1,tkin_pp2, u_pp1,u_pp2
    real*8 :: dir(5),exc(5),r_now(5)


    ep1=p1_4(1)
    p1=sqrt(sum(p1_4(2:4)**2))
    p1_4(1)=sqrt(p1**2+xmn**2)
    ep2=p2_4(1)
    p2=sqrt(sum(p2_4(2:4)**2))   
    p2_4(1)=sqrt(p2**2+xmn**2)
    pp1=sqrt(sum(pp1_4(2:4)**2))   
    if(sqrt(sum(pp2_4(2:4)**2)).lt.xpf) then
        r_now=0.0d0
        return
    endif


    !......define constants and ff
    q2=q_4(1)**2-sum(q_4(2:4)**2)
    gep=1.0d0/(1.0d0-q2/lsq)**2 
    cv3=fstar/(1.0d0-q2/lsq)**2/(1.0d0-q2/4.0d0/lsq)*sqrt(3.0d0/2.0d0)
    ca5=0.0d0!1.20d0/(1.0d0-q2/xma2)**2/(1.0d0-q2/3.0d0/xma2)*sqrt(3.0d0/2.0d0)
    rho=xpf**3/(1.5d0*pi**2)

    tkin_pp1=pp1_4(1)-xmn
    tkin_pp2=pp2_4(1)-xmn
    u_pp1=0.0d0
    u_pp2=0.0d0
    !if(i_fsi.eq.1.and.(tkin_pp1.lt.kin(npot)).and.(tkin_pp1.gt.kin(1))) call interpolint(kin,pot,npot,tkin_pp1,u_pp1,1)
    !if(i_fsi.eq.1.and.(tkin_pp2.lt.kin(npot)).and.(tkin_pp2.gt.kin(1))) call interpolint(kin,pot,npot,tkin_pp2,u_pp2,1)

    !...I define w_tilde here...we might want to change it
    if(i_fg.eq.1) then
        q_4(1)=q_4(1)-40.0d0
    else
        ! q_4(1)=w-p1_4(1)-p2_4(1)-ep(ie1)+xmn-ep(ie2)+xmn+60.0d0!-u_pp1-u_pp2  
        q_4(1)=q_4(1)-p1_4(1)-p2_4(1)-ep1+xmn-ep2+xmn+120.0d0  
    endif

    !...delta function
    arg=q_4(1)+p1_4(1)+p2_4(1)-pp1_4(1)-pp2_4(1)
    delta_w=fdelta(arg,eps)
    !...define pion momenta
    k1_4(:)=pp1_4(:)-p1_4(:)
    k2_4(:)=q_4(:)-k1_4(:)
    k1e_4(:)=pp2_4(:)-p1_4(:)
    k2e_4(:)=q_4(:)-k1e_4(:)

    !.......currents
    call current_init_two_body(p1_4,p2_4,pp1_4,pp2_4,q_4,k1_4,k2_4,1)      
    call define_spinors()
    call det_Jpi(gep)
    call det_JpiJpi(r_cc_pi,r_cl_pi,r_ll_pi,r_t_pi,r_tp_pi)
    call det_JaJb_JcJd(cv3,ca5,np_del,pdel,pot_del)
    call det_JaJc_dir(r_cc_del,r_cl_del,r_ll_del,r_t_del,r_tp_del)
    call det_JpiJaJb(r_cc_int,r_cl_int,r_ll_int,r_t_int,r_tp_int)

    dir(1)=r_cc_pi+2.0d0*(r_cc_del+r_cc_int)
    dir(2)=r_cl_pi+2.0d0*(r_cl_del+r_cl_int)
    dir(3)=r_ll_pi+2.0d0*(r_ll_del+r_ll_int)
    dir(4)=r_t_pi+2.0d0*(r_t_del+r_t_int)
    dir(5)=r_tp_pi+2.0d0*(r_tp_del+r_tp_int)

    call current_init_two_body(p1_4,p2_4,pp2_4,pp1_4,q_4,k1e_4,k2e_4,2)
    call det_JaJb_JcJd(cv3,ca5,np_del,pdel,pot_del)
    call det_JaJc_exc(r_cc_del,r_cl_del,r_ll_del,r_t_del,r_tp_del)
    call det_JpiJaJb_exc(r_cc_int,r_cl_int,r_ll_int,r_t_int,r_tp_int)

    exc(1)=2.0d0*(r_cc_del+r_cc_int)
    exc(2)=2.0d0*(r_cl_del+r_cl_int)
    exc(3)=2.0d0*(r_ll_del+r_ll_int)
    exc(4)=2.0d0*(r_t_del+r_t_int)
    exc(5)=2.0d0*(r_tp_del+r_tp_int)

    r_now(:) = dp1*dp2*p1**2*p2**2/(2.0d0*pi)**8*(dir(:)-exc(:))* &
        delta_w*pp1**2/rho*dble(xA)/2.0d0/2.0d0 ! /2.0d0 for the electromagnetic piece

    return
end subroutine two_body 

end module hadronic_response
