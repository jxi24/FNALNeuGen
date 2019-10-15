!subroutine form_factor(x, y)
!    integer, parameter :: n = 4
!    real, INTENT(IN) :: x
!    real, INTENT(OUT) :: y(n)
!    external py_form_factor
!    !f2py intent(callback, hide) py_form_factor
!    !f2py intent(in) :: x
!    !f2py intent(hide) :: n
!    !f2py intent(out), depend(n) :: y
!
!    print*, x
!
!    call py_form_factor(x, y, n)
!
!    print*, y
!
!    return
!end subroutine form_factor  

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
