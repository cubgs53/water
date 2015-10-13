#ifndef CENTRAL2D_H
#define CENTRAL2D_H

#include "stepper.h"

#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>

//ldoc on
/**
 * # Jiang-Tadmor central difference scheme
 *
 * [Jiang and Tadmor][jt] proposed a high-resolution finite difference
 * scheme for solving hyperbolic PDE systems in two space dimensions.
 * The method is particularly attractive because, unlike many other
 * methods in this space, it does not require that we write any
 * solvers for problems with special initial data (so-called Riemann
 * problems), nor even that we compute Jacobians of the flux
 * functions.
 *
 * While this code is based loosely on the Fortran code at the end of
 * Jiang and Tadmor's paper, we've written the current code to be
 * physics-agnostic (rather than hardwiring it to the shallow water
 * equations -- or the Euler equations in the Jiang-Tadmor paper).
 * If you're interested in the Euler equations, feel free to add your
 * own physics class to support them!
 *
 * [jt]: http://www.cscamm.umd.edu/tadmor/pub/central-schemes/Jiang-Tadmor.SISSC-98.pdf
 *
 * ## Staggered grids
 *
 * The Jiang-Tadmor scheme works by alternating between a main grid
 * and a staggered grid offset by half a step in each direction.
 * Understanding this is important, particularly if you want to apply
 * a domain decomposition method and batch time steps between
 * synchronization barriers in your parallel code!
 *
 * In even-numbered steps, the entry `u(i,j)` in the array of solution
 * values represents the average value of a cell centered at a point
 * $(x_i,y_j)$.  At the following odd-numbered step, the same entry
 * represents values for a cell centered at $(x_i + \Delta x/2, y_j +
 * \Delta y/2)$.  However, whenever we run a simulation, we always take
 * an even number of steps, so that outside the solver we can just think
 * about values on the main grid.  If `uold` and `unew` represent the
 * information at two successive *even* time steps (i.e. they represent
 * data on the same grid), then `unew(i,j)` depends indirectly on
 * `u(p,q)` for $i-3 \leq p \leq i+3$ and $j-3 \leq q \leq j+3$.
 *
 * We currently manage this implicitly: the arrays at even time steps
 * represent cell values on the main grid, and arrays at odd steps
 * represent cell values on the staggered grid.  Our main `run`
 * function always takes an even number of time steps to ensure we end
 * up on the primary grid.
 *
 * ## MinMod limiter
 *
 * Numerical methods for solving nonlinear wave equations are
 * complicated by the fact that even with smooth initial data, a
 * nonlinear wave can develop discontinuities (shocks) in finite time.
 *
 * This makes for interesting analysis, since a "strong" solution
 * that satisfies the differential equation no longer makes sense at
 * a shock -- instead, we have to come up with some mathematically
 * and physically reasonable definition of a "weak" solution that
 * satisfies the PDE away from the shock and satisfies some other
 * condition (an entropy condition) at the shock.
 *
 * The presence of shocks also makes for interesting *numerical*
 * analysis, because we need to be careful about employing numerical
 * differentiation formulas that sample a discontinuous function at
 * points on different sides of a shock.  Using such formulas naively
 * usually causes the numerical method to become unstable.  A better
 * method -- better even in the absence of shocks! -- is to consider
 * multiple numerical differentiation formulas and use the highest
 * order one that "looks reasonable" in the sense that it doesn't
 * predict wildly larger slopes than the others.  Because these
 * combined formulas *limit* the wild behavior of derivative estimates
 * across a shock, we call them *limiters*.  With an appropriate limiter,
 * we can construct methods that have high-order accuracy away from shocks
 * and are at least first-order accurate close to a shock.  These are
 * sometimes called *high-resolution* methods.
 *
 * The MinMod (minimum modulus) limiter is one example of a limiter.
 * The MinMod limiter estimates the slope through points $f_-, f_0, f_+$
 * (with the step $h$ scaled to 1) by
 * $$
 *   f' = \operatorname{minmod}((f_+-f_-)/2, \theta(f_+-f_0), \theta(f_0-f_-))
 * $$
 * where the minmod function returns the argument with smallest absolute
 * value if all arguments have the same sign, and zero otherwise.
 * Common choices of $\theta$ are $\theta = 1.0$ and $\theta = 2.0$.
 *
 * The minmod limiter *looks* like it should be expensive to computer,
 * since superficially it seems to require a number of branches.
 * We do something a little tricky, getting rid of the condition
 * on the sign of the arguments using the `copysign` instruction.
 * If the compiler does the "right" thing with `max` and `min`
 * for floating point arguments (translating them to branch-free
 * intrinsic operations), this implementation should be relatively fast.
 *
 * There are many other potential choices of limiters as well.  We'll
 * stick with this one for the code, but you should feel free to
 * experiment with others if you know what you're doing and think it
 * will improve performance or accuracy.
 *
 * ## Interface
 *
 * We want a clean separation between the physics, the solver,
 * and the auxiliary limiter methods used by the solver.  At the same
 * time, we don't want to pay the overhead (mostly in terms of lost
 * optimization opportunities) for calling across an abstraction
 * barrier in the inner loops of our solver.  We can get around this
 * in C++ by providing the solver with *template arguments*, resolved
 * at compile time, that describe separate classes to implement the
 * physics and the limiter.
 *
 * The `Central2D` solver class takes the template arguments `Physics`
 * that defines:
 *
 *  - A type for numerical data (`real`)
 *  - A type for solution and flux vectors in each cell (`vec`)
 *  - A flux computation function (`flux(vec& F, vec& G, const vec& U)`)
 *  - A wave speed computation function
 *    (`wave_speed(real& cx, real& cy, const vec& U)`).
 *
 * The solver keeps arrays for the solution, flux values, derivatives
 * of the solution and the fluxes, and the solution at the next time
 * point.  We use the C++ `vector` class to manage storage for these
 * arrays; but since we want to think of them as 2D arrays, we also
 * provide convenience functions to access them with multiple indices
 * (though we maintain C-style 0-based indexing).  The internal arrays
 * are padded with ghost cells; the ghost cell in the lower left corner
 * of the domain has index (0,0).
 */

class Central2D {
public:
    typedef float real;
    typedef void (*flux_t)(real* FU, real* GU, const real* U,
                           int ncell, int field_stride);
    typedef void (*speed_t)(real* cxy, const real* U,
                            int ncell, int field_stride);

    const int nx, ny;          // Number of (non-ghost) cells in x/y
    const int nfield;          // Number of fields
    
    Central2D(real w, real h,     // Domain width / height
              int nx, int ny,     // Number of cells in x/y (without ghosts)
              int nfield,         // Number of field
              flux_t flux,        // Flux computation
              speed_t speed,      // Speed computation
              real cfl = 0.45) :  // Max allowed CFL number
        nx(nx), ny(ny), nfield(nfield), 
        nx_all(nx + 2*nghost),
        ny_all(ny + 2*nghost),
        dx(w/nx), dy(h/ny),
        flux(flux), speed(speed),
        cfl(cfl),
        u_ (nfield * nx_all * ny_all),
        f_ (nfield * nx_all * ny_all),
        g_ (nfield * nx_all * ny_all),
        ux_(nfield * nx_all * ny_all),
        uy_(nfield * nx_all * ny_all),
        fx_(nfield * nx_all * ny_all),
        gy_(nfield * nx_all * ny_all),
        v_ (nfield * nx_all * ny_all) {}

    // Advance from time 0 to time tfinal
    void run(real tfinal);

    // Diagnostics
    void solution_check();

    // Array size accessors
    int xsize() const { return nx; }
    int ysize() const { return ny; }

    // Dimension accessors
    real get_dx() const { return dx; }
    real get_dy() const { return dy; }

    // Read / write elements of simulation state
    real& operator()(int k, int i, int j) {
        return u_[offset(k,i+nghost,j+nghost)];
    }
    real operator()(int k, int i, int j) const {
        return u_[offset(k,i+nghost,j+nghost)];
    }

private:
    static constexpr int nghost = 3;   // Number of ghost cells

    const int nx_all, ny_all;  // Total cells in x/y (including ghost)
    const real dx, dy;         // Cell size in x/y
    const flux_t flux;         // Flux function pointer
    const speed_t speed;       // Speed function pointer
    const real cfl;            // Allowed CFL number

    std::vector<real> u_;            // Solution values
    std::vector<real> f_;            // Fluxes in x
    std::vector<real> g_;            // Fluxes in y
    std::vector<real> ux_;           // x differences of u
    std::vector<real> uy_;           // y differences of u
    std::vector<real> fx_;           // x differences of f
    std::vector<real> gy_;           // y differences of g
    std::vector<real> v_;            // Solution values at next step

    // Array accessor functions

    int offset(int k, int ix, int iy) const { return (k*ny_all+iy)*nx_all+ix; }

    real& u(int k, int ix, int iy)    { return u_[offset(k,ix,iy)]; }
    real& v(int k, int ix, int iy)    { return v_[offset(k,ix,iy)]; }
    real& f(int k, int ix, int iy)    { return f_[offset(k,ix,iy)]; }
    real& g(int k, int ix, int iy)    { return g_[offset(k,ix,iy)]; }

    real& ux(int k, int ix, int iy)   { return ux_[offset(k,ix,iy)]; }
    real& uy(int k, int ix, int iy)   { return uy_[offset(k,ix,iy)]; }
    real& fx(int k, int ix, int iy)   { return fx_[offset(k,ix,iy)]; }
    real& gy(int k, int ix, int iy)   { return gy_[offset(k,ix,iy)]; }

    // Wrapped accessor (periodic BC)
    int ioffset(int k, int ix, int iy) {
        return offset(k,
                      (ix+nx-nghost) % nx + nghost,
                      (iy+ny-nghost) % ny + nghost );
    }

    real& uwrap(int k, int ix, int iy)  { return u_[ioffset(k,ix,iy)]; }

    // Stages of the main algorithm
    void apply_periodic();
    void compute_step(int io, real dt);

};


/**
 * ## Time stepper implementation
 *
 * ### Boundary conditions
 *
 * In finite volume methods, boundary conditions are typically applied by
 * setting appropriate values in ghost cells.  For our framework, we will
 * apply periodic boundary conditions; that is, waves that exit one side
 * of the domain will enter from the other side.
 *
 * We apply the conditions by assuming that the cells with coordinates
 * `nghost <= ix <= nx+nghost` and `nghost <= iy <= ny+nghost` are
 * "canonical", and setting the values for all other cells `(ix,iy)`
 * to the corresponding canonical values `(ix+p*nx,iy+q*ny)` for some
 * integers `p` and `q`.
 */

void Central2D::apply_periodic()
{
    // Copy data between right and left boundaries
    for (int k = 0; k < nfield; ++k) {
        for (int iy = 0; iy < ny_all; ++iy)
            for (int ix = 0; ix < nghost; ++ix) {
                u(k,ix,          iy) = uwrap(k,ix,          iy);
                u(k,nx+nghost+ix,iy) = uwrap(k,nx+nghost+ix,iy);
            }

        // Copy data between top and bottom boundaries
        for (int iy = 0; iy < nghost; ++iy)
            for (int ix = 0; ix < nx_all; ++ix) {
                u(k,ix,          iy) = uwrap(k,ix,          iy);
                u(k,ix,ny+nghost+iy) = uwrap(k,ix,ny+nghost+iy);
            }
    }
}


/**
 * ### Advancing a time step
 *
 * Take one step of the numerical scheme.  This consists of two pieces:
 * a first-order corrector computed at a half time step, which is used
 * to obtain new $F$ and $G$ values; and a corrector step that computes
 * the solution at the full step.  For full details, we refer to the
 * [Jiang and Tadmor paper][jt].
 *
 * The `compute_step` function takes two arguments: the `io` flag
 * which is the time step modulo 2 (0 if even, 1 if odd); and the `dt`
 * flag, which actually determines the time step length.  We need
 * to know the even-vs-odd distinction because the Jiang-Tadmor
 * scheme alternates between a primary grid (on even steps) and a
 * staggered grid (on odd steps).  This means that the data at $(i,j)$
 * in an even step and the data at $(i,j)$ in an odd step represent
 * values at different locations in space, offset by half a space step
 * in each direction.  Every other step, we shift things back by one
 * mesh cell in each direction, essentially resetting to the primary
 * indexing scheme.
 */

void Central2D::compute_step(int io, real dt)
{
    real dtcdx2 = 0.5 * dt / dx;
    real dtcdy2 = 0.5 * dt / dy;

    flux(&f_[0], &g_[0], &u_[0],
         nx_all * ny_all, nx_all * ny_all);

    central2d_derivs(&ux_[0], &uy_[0],
                     &fx_[0], &gy_[0],
                     &u_[0], &f_[0], &g_[0],
                     nx_all, ny_all, nfield);

    central2d_predict(&v_[0], &u_[0], &fx_[0], &gy_[0],
                      dtcdx2, dtcdy2, nx_all, ny_all, nfield);

    // Flux values of f and g at half step
    for (int iy = 1; iy < ny_all-1; ++iy) {
        int jj = offset(0,1,iy);
        flux(&f_[jj], &g_[jj], &v_[jj],
             nx_all-2, nx_all * ny_all);
    }

    central2d_correct(&v_[0], &u_[0], &ux_[0], &uy_[0], &f_[0], &g_[0],
                      dtcdx2, dtcdy2,
                      nghost-io, nx+nghost-io,
                      nghost-io, ny+nghost-io,
                      nx_all, ny_all, nfield);

    // Copy from v storage back to main grid
    for (int k = 0; k < nfield; ++k)
        memcpy(&u(k,nghost,nghost),
               &v(k,nghost-io,nghost-io),
               ny * nx_all * sizeof(float));
}


/**
 * ### Advance time
 *
 * The `run` method advances from time 0 (initial conditions) to time
 * `tfinal`.  Note that `run` can be called repeatedly; for example,
 * we might want to advance for a period of time, write out a picture,
 * advance more, and write another picture.  In this sense, `tfinal`
 * should be interpreted as an offset from the time represented by
 * the simulator at the start of the call, rather than as an absolute time.
 *
 * We always take an even number of steps so that the solution
 * at the end lives on the main grid instead of the staggered grid.
 */

void Central2D::run(real tfinal)
{
    bool done = false;
    real t = 0;
    while (!done) {
        real dt;
        for (int io = 0; io < 2; ++io) {
            real cxy[2] = {1.0e-15f, 1.0e-15f};
            apply_periodic();
            speed(cxy, &u_[0], nx_all * ny_all, nx_all * ny_all);
            if (io == 0) {
                dt = cfl / std::max(cxy[0]/dx, cxy[1]/dy);
                if (t + 2*dt >= tfinal) {
                    dt = (tfinal-t)/2;
                    done = true;
                }
            }
            compute_step(io, dt);
            t += dt;
        }
    }
}

/**
 * ### Diagnostics
 *
 * The numerical method is supposed to preserve (up to rounding
 * errors) the total volume of water in the domain and the total
 * momentum.  Ideally, we should also not see negative water heights,
 * since that will cause the system of equations to blow up.  For
 * debugging convenience, we'll plan to periodically print diagnostic
 * information about these conserved quantities (and about the range
 * of water heights).
 */

void Central2D::solution_check()
{
    using namespace std;
    real h_sum = 0, hu_sum = 0, hv_sum = 0;
    real hmin = u(0,nghost,nghost);
    real hmax = hmin;
    for (int j = nghost; j < ny+nghost; ++j)
        for (int i = nghost; i < nx+nghost; ++i) {
            real h = u(0,i,j);
            h_sum += h;
            hu_sum += u(1,i,j);
            hv_sum += u(2,i,j);
            hmax = max(h, hmax);
            hmin = min(h, hmin);
        }
    real cell_area = dx*dy;
    h_sum *= cell_area;
    hu_sum *= cell_area;
    hv_sum *= cell_area;
    printf("-\n  Volume: %g\n  Momentum: (%g, %g)\n  Range: [%g, %g]\n",
           h_sum, hu_sum, hv_sum, hmin, hmax);
    assert(hmin > 0);
}

//ldoc off
#endif /* CENTRAL2D_H*/
