#ifndef CENTRAL2D_H
#define CENTRAL2D_H

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
 * The `Central2D` solver class takes two template arguments:
 * `Physics` and `Limiter`.  For `Physics`, we expect the name of a class
 * that defines:
 * 
 *  - A type for numerical data (`real`)
 *  - A type for solution and flux vectors in each cell (`vec`)
 *  - A flux computation function (`flux(vec& F, vec& G, const vec& U)`)
 *  - A wave speed computation function 
 *    (`wave_speed(real& cx, real& cy, const vec& U)`).
 * 
 * The `Limiter` argument is a type with a static function `limdiff`
 * with the signature
 * 
 *         limdiff(fm, f0, fp)
 * 
 * The semantics are that `fm`, `f0`, and `fp` are three successive
 * grid points in some direction, and the function returns an approximate
 * (scaled) derivative value from these points.
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

template <class Physics, class Limiter>
class Central2D {
public:
    typedef typename Physics::real real;
    typedef typename Physics::vec  vec;

    Central2D(real w, real h,     // Domain width / height
              int nx, int ny,     // Number of cells in x/y (without ghosts)
              real cfl = 0.45) :  // Max allowed CFL number
        nx(nx), ny(ny),
        nx_all(nx + 2*nghost),
        ny_all(ny + 2*nghost),
        dx(w/nx), dy(h/ny),
        cfl(cfl),
        // u_ (nx_all * ny_all),
        // f_ (nx_all * ny_all),
        // g_ (nx_all * ny_all),
        // ux_(nx_all * ny_all),
        // uy_(nx_all * ny_all),
        // fx_(nx_all * ny_all),
        // gy_(nx_all * ny_all),
        // v_ (nx_all * ny_all),

        u0_ (nx_all * ny_all),
        u1_ (nx_all * ny_all),
        u2_ (nx_all * ny_all),

        f0_ (nx_all * ny_all),
        f1_ (nx_all * ny_all),
        f2_ (nx_all * ny_all),

        g0_ (nx_all * ny_all),
        g1_ (nx_all * ny_all),
        g2_ (nx_all * ny_all),

        ux0_(nx_all * ny_all),
        ux1_(nx_all * ny_all),
        ux2_(nx_all * ny_all),

        uy0_(nx_all * ny_all),
        uy1_(nx_all * ny_all),
        uy2_(nx_all * ny_all),

        fx0_(nx_all * ny_all),
        fx1_(nx_all * ny_all),
        fx2_(nx_all * ny_all),

        gy0_(nx_all * ny_all),
        gy1_(nx_all * ny_all),
        gy2_(nx_all * ny_all),

        v0_(nx_all * ny_all),
        v1_(nx_all * ny_all),
        v2_(nx_all * ny_all) {}

    // Advance from time 0 to time tfinal
    void run(real tfinal);

    // Call f(Uxy, x, y) at each cell center to set initial conditions
    template <typename F>
    void init(F f);

    // Diagnostics
    void solution_check();

    // Array size accessors
    int xsize() const { return nx; }
    int ysize() const { return ny; }

    // Read / write elements of simulation state
    // vec&       operator()(int i, int j) {
    //     return u_[offset(i+nghost,j+nghost)];
    // }

    // const vec& operator()(int i, int j) const {
    //     return u_[offset(i+nghost,j+nghost)];
    // }

    // Read / write elements of simulation state
    real&       operator()(int i, int j) {
        return u0_[offset(i+nghost,j+nghost)];
    }

    const real& operator()(int i, int j) const {
        return u0_[offset(i+nghost,j+nghost)];
    }

private:
    static constexpr int nghost = 3;   // Number of ghost cells

    const int nx, ny;          // Number of (non-ghost) cells in x/y
    const int nx_all, ny_all;  // Total cells in x/y (including ghost)
    const real dx, dy;         // Cell size in x/y
    const real cfl;            // Allowed CFL number

    // std::vector<vec> u_;            // Solution values
    // std::vector<vec> f_;            // Fluxes in x
    // std::vector<vec> g_;            // Fluxes in y
    // std::vector<vec> ux_;           // x differences of u
    // std::vector<vec> uy_;           // y differences of u
    // std::vector<vec> fx_;           // x differences of f
    // std::vector<vec> gy_;           // y differences of g
    // std::vector<vec> v_;            // Solution values at next step

    std::vector<real> u0_;            // Solution values
    std::vector<real> f0_;            // Fluxes in x
    std::vector<real> g0_;            // Fluxes in y
    std::vector<real> ux0_;           // x differences of u
    std::vector<real> uy0_;           // y differences of u
    std::vector<real> fx0_;           // x differences of f
    std::vector<real> gy0_;           // y differences of g
    std::vector<real> v0_;            // Solution values at next step

    std::vector<real> u1_;            // Solution values
    std::vector<real> f1_;            // Fluxes in x
    std::vector<real> g1_;            // Fluxes in y
    std::vector<real> ux1_;           // x differences of u
    std::vector<real> uy1_;           // y differences of u
    std::vector<real> fx1_;           // x differences of f
    std::vector<real> gy1_;           // y differences of g
    std::vector<real> v1_;            // Solution values at next step

    std::vector<real> u2_;            // Solution values
    std::vector<real> f2_;            // Fluxes in x
    std::vector<real> g2_;            // Fluxes in y
    std::vector<real> ux2_;           // x differences of u
    std::vector<real> uy2_;           // y differences of u
    std::vector<real> fx2_;           // x differences of f
    std::vector<real> gy2_;           // y differences of g
    std::vector<real> v2_;            // Solution values at next step

    // Array accessor functions

    int offset(int ix, int iy) const { return iy*nx_all+ix; }

    // vec& u(int ix, int iy)    { return u_[offset(ix,iy)]; }
    // vec& v(int ix, int iy)    { return v_[offset(ix,iy)]; }
    // vec& f(int ix, int iy)    { return f_[offset(ix,iy)]; }
    // vec& g(int ix, int iy)    { return g_[offset(ix,iy)]; }

    // vec& ux(int ix, int iy)   { return ux_[offset(ix,iy)]; }
    // vec& uy(int ix, int iy)   { return uy_[offset(ix,iy)]; }
    // vec& fx(int ix, int iy)   { return fx_[offset(ix,iy)]; }
    // vec& gy(int ix, int iy)   { return gy_[offset(ix,iy)]; }

    real& u0(int ix, int iy)    { return u0_[offset(ix,iy)]; }
    real& v0(int ix, int iy)    { return v0_[offset(ix,iy)]; }
    real& f0(int ix, int iy)    { return f0_[offset(ix,iy)]; }
    real& g0(int ix, int iy)    { return g0_[offset(ix,iy)]; }

    real& ux0(int ix, int iy)   { return ux0_[offset(ix,iy)]; }
    real& uy0(int ix, int iy)   { return uy0_[offset(ix,iy)]; }
    real& fx0(int ix, int iy)   { return fx0_[offset(ix,iy)]; }
    real& gy0(int ix, int iy)   { return gy0_[offset(ix,iy)]; }


    real& u1(int ix, int iy)    { return u1_[offset(ix,iy)]; }
    real& v1(int ix, int iy)    { return v1_[offset(ix,iy)]; }
    real& f1(int ix, int iy)    { return f1_[offset(ix,iy)]; }
    real& g1(int ix, int iy)    { return g1_[offset(ix,iy)]; }

    real& ux1(int ix, int iy)   { return ux1_[offset(ix,iy)]; }
    real& uy1(int ix, int iy)   { return uy1_[offset(ix,iy)]; }
    real& fx1(int ix, int iy)   { return fx1_[offset(ix,iy)]; }
    real& gy1(int ix, int iy)   { return gy1_[offset(ix,iy)]; }


    real& u2(int ix, int iy)    { return u2_[offset(ix,iy)]; }
    real& v2(int ix, int iy)    { return v2_[offset(ix,iy)]; }
    real& f2(int ix, int iy)    { return f2_[offset(ix,iy)]; }
    real& g2(int ix, int iy)    { return g2_[offset(ix,iy)]; }

    real& ux2(int ix, int iy)   { return ux2_[offset(ix,iy)]; }
    real& uy2(int ix, int iy)   { return uy2_[offset(ix,iy)]; }
    real& fx2(int ix, int iy)   { return fx2_[offset(ix,iy)]; }
    real& gy2(int ix, int iy)   { return gy2_[offset(ix,iy)]; }

    // Wrapped accessor (periodic BC)
    int ioffset(int ix, int iy) {
        return offset( (ix+nx-nghost) % nx + nghost,
                       (iy+ny-nghost) % ny + nghost );
    }

    real& uwrap0(int ix, int iy)  { return u0_[ioffset(ix,iy)]; }
    real& uwrap1(int ix, int iy)  { return u1_[ioffset(ix,iy)]; }
    real& uwrap2(int ix, int iy)  { return u2_[ioffset(ix,iy)]; }
    // vec& uwrap(int ix, int iy)  { return u_[ioffset(ix,iy)]; }

    // Apply limiter to all components in a vector
    static void limdiff(real& du, const real& um, const real& u0, const real& up) {
        du = Limiter::limdiff(um, u0, up);
    }
    // static void limdiff(vec& du, const vec& um, const vec& u0, const vec& up) {
    //     for (int m = 0; m < du.size(); ++m)
    //         du[m] = Limiter::limdiff(um[m], u0[m], up[m]);
    // }

    // Stages of the main algorithm
    void apply_periodic();
    void compute_fg_speeds(real& cx, real& cy);
    void limited_derivs();
    void compute_step(int io, real dt);

};


/**
 * ## Initialization
 * 
 * Before starting the simulation, we need to be able to set the
 * initial conditions.  The `init` function does exactly this by
 * running a callback function at the center of each cell in order
 * to initialize the cell $U$ value.  For the purposes of this function,
 * cell $(i,j)$ is the subdomain 
 * $[i \Delta x, (i+1) \Delta x] \times [j \Delta y, (j+1) \Delta y]$.
 */

template <class Physics, class Limiter>
template <typename F>
void Central2D<Physics, Limiter>::init(F f)
{
    for (int iy = 0; iy < ny; ++iy) {
        #pragma ivdep
        for (int ix = 0; ix < nx; ++ix) {
            f(u0(nghost+ix,nghost+iy), u1(nghost+ix,nghost+iy), u2(nghost+ix,nghost+iy),
              (ix+0.5)*dx, (iy+0.5)*dy);
            // f(u(nghost+ix,nghost+iy), (ix+0.5)*dx, (iy+0.5)*dy);
        }
    }
}

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

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::apply_periodic()
{
    // Copy data between right and left boundaries
    for (int iy = 0; iy < ny_all; ++iy)
        #pragma ivdep
        for (int ix = 0; ix < nghost; ++ix) {
            // u(ix,          iy) = uwrap(ix,          iy);
            // u(nx+nghost+ix,iy) = uwrap(nx+nghost+ix,iy);
            u0(ix,          iy) = uwrap0(ix,          iy);
            u0(nx+nghost+ix,iy) = uwrap0(nx+nghost+ix,iy);
            u1(ix,          iy) = uwrap1(ix,          iy);
            u1(nx+nghost+ix,iy) = uwrap1(nx+nghost+ix,iy);
            u2(ix,          iy) = uwrap2(ix,          iy);
            u2(nx+nghost+ix,iy) = uwrap2(nx+nghost+ix,iy);
        }

    // Copy data between top and bottom boundaries
    for (int ix = 0; ix < nx_all; ++ix)
        #pragma ivdep
        for (int iy = 0; iy < nghost; ++iy) {
            u0(ix,          iy) = uwrap0(ix,          iy);
            u0(ix,ny+nghost+iy) = uwrap0(ix,ny+nghost+iy);
            u1(ix,          iy) = uwrap1(ix,          iy);
            u1(ix,ny+nghost+iy) = uwrap1(ix,ny+nghost+iy);
            u2(ix,          iy) = uwrap2(ix,          iy);
            u2(ix,ny+nghost+iy) = uwrap2(ix,ny+nghost+iy);
        }
}


/**
 * ### Initial flux and speed computations
 * 
 * At the start of each time step, we need the flux values at
 * cell centers (to advance the numerical method) and a bound
 * on the wave speeds in the $x$ and $y$ directions (so that
 * we can choose a time step that respects the specified upper
 * bound on the CFL number).
 */

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::compute_fg_speeds(real& cx_, real& cy_)
{
    using namespace std;
    real cx = 1.0e-15;
    real cy = 1.0e-15;
    for (int iy = 0; iy < ny_all; ++iy)
        #pragma ivdep
        for (int ix = 0; ix < nx_all; ++ix) {
            real cell_cx, cell_cy;
            Physics::flux(f0(ix,iy), g0(ix,iy), u0(ix,iy),
                          f1(ix,iy), g1(ix,iy), u1(ix,iy),
                          f2(ix,iy), g2(ix,iy), u2(ix,iy));
            Physics::wave_speed(cell_cx, cell_cy,
                                u0(ix,iy), u1(ix,iy), u2(ix,iy));
            cx = max(cx, cell_cx);
            cy = max(cy, cell_cy);
        }
    cx_ = cx;
    cy_ = cy;
}

/**
 * ### Derivatives with limiters
 * 
 * In order to advance the time step, we also need to estimate
 * derivatives of the fluxes and the solution values at each cell.
 * In order to maintain stability, we apply a limiter here.
 */

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::limited_derivs()
{
    //x derivatives
    // for (int iy = 1; iy < ny_all-1; ++iy) {
    //     #pragma ivdep
    //     for (int ix = 1; ix < nx_all-1; ++ix) {
    //         limdiff( ux0(ix,iy), u0(ix-1,iy), u0(ix,iy), u0(ix+1,iy) );
    //     }
    // }
    // for (int iy = 1; iy < ny_all-1; ++iy) {
    //     #pragma ivdep
    //     for (int ix = 1; ix < nx_all-1; ++ix) {
    //         limdiff( fx0(ix,iy), f0(ix-1,iy), f0(ix,iy), f0(ix+1,iy) );
    //     }
    // }
    // for (int iy = 1; iy < ny_all-1; ++iy) {
    //     #pragma ivdep
    //     for (int ix = 1; ix < nx_all-1; ++ix) {
    //         limdiff( ux1(ix,iy), u1(ix-1,iy), u1(ix,iy), u1(ix+1,iy) );
    //     }
    // }
    // for (int iy = 1; iy < ny_all-1; ++iy) {
    //     #pragma ivdep
    //     for (int ix = 1; ix < nx_all-1; ++ix) {
    //         limdiff( fx1(ix,iy), f1(ix-1,iy), f1(ix,iy), f1(ix+1,iy) );
    //     }
    // }
    // for (int iy = 1; iy < ny_all-1; ++iy) {
    //     #pragma ivdep
    //     for (int ix = 1; ix < nx_all-1; ++ix) {
    //         limdiff( ux2(ix,iy), u2(ix-1,iy), u2(ix,iy), u2(ix+1,iy) );
    //     }
    // }
    // for (int iy = 1; iy < ny_all-1; ++iy) {
    //     #pragma ivdep
    //     for (int ix = 1; ix < nx_all-1; ++ix) {
    //         limdiff( fx2(ix,iy), f2(ix-1,iy), f2(ix,iy), f2(ix+1,iy) );
    //     }
    // }

    // //y derivatives
    // for (int iy = 1; iy < ny_all-1; ++iy) {
    //     #pragma ivdep
    //     for (int ix = 1; ix < nx_all-1; ++ix) {
    //         limdiff( gy0(ix,iy), g0(ix,iy-1), g0(ix,iy), g0(ix,iy+1) );
    //     }
    // }
    // for (int iy = 1; iy < ny_all-1; ++iy) {
    //     #pragma ivdep
    //     for (int ix = 1; ix < nx_all-1; ++ix) {
    //         limdiff( uy1(ix,iy), u1(ix,iy-1), u1(ix,iy), u1(ix,iy+1) );
    //     }
    // }
    // for (int iy = 1; iy < ny_all-1; ++iy) {
    //     #pragma ivdep
    //     for (int ix = 1; ix < nx_all-1; ++ix) {
    //         limdiff( gy1(ix,iy), g1(ix,iy-1), g1(ix,iy), g1(ix,iy+1) );
    //     }
    // }
    // for (int iy = 1; iy < ny_all-1; ++iy) {
    //     #pragma ivdep
    //     for (int ix = 1; ix < nx_all-1; ++ix) {
    //         limdiff( uy0(ix,iy), u0(ix,iy-1), u0(ix,iy), u0(ix,iy+1) );
    //     }
    // }
    // for (int iy = 1; iy < ny_all-1; ++iy) {
    //     #pragma ivdep
    //     for (int ix = 1; ix < nx_all-1; ++ix) {
    //         limdiff( uy2(ix,iy), u2(ix,iy-1), u2(ix,iy), u2(ix,iy+1) );
    //     }
    // }
    // for (int iy = 1; iy < ny_all-1; ++iy) {
    //     #pragma ivdep
    //     for (int ix = 1; ix < nx_all-1; ++ix) {
    //         limdiff( gy2(ix,iy), g2(ix,iy-1), g2(ix,iy), g2(ix,iy+1) );
    //     }
    // }
    for (int iy = 1; iy < ny_all-1; ++iy) {
        #pragma ivdep
        for (int ix = 1; ix < nx_all-1; ++ix) {
            // x derivs
            limdiff( ux0(ix,iy), u0(ix-1,iy), u0(ix,iy), u0(ix+1,iy) );
            limdiff( fx0(ix,iy), f0(ix-1,iy), f0(ix,iy), f0(ix+1,iy) );
            limdiff( ux1(ix,iy), u1(ix-1,iy), u1(ix,iy), u1(ix+1,iy) );
            limdiff( fx1(ix,iy), f1(ix-1,iy), f1(ix,iy), f1(ix+1,iy) );
            limdiff( ux2(ix,iy), u2(ix-1,iy), u2(ix,iy), u2(ix+1,iy) );
            limdiff( fx2(ix,iy), f2(ix-1,iy), f2(ix,iy), f2(ix+1,iy) );

            // y derivs
            limdiff( uy0(ix,iy), u0(ix,iy-1), u0(ix,iy), u0(ix,iy+1) );
            limdiff( gy0(ix,iy), g0(ix,iy-1), g0(ix,iy), g0(ix,iy+1) );
            limdiff( uy1(ix,iy), u1(ix,iy-1), u1(ix,iy), u1(ix,iy+1) );
            limdiff( gy1(ix,iy), g1(ix,iy-1), g1(ix,iy), g1(ix,iy+1) );
            limdiff( uy2(ix,iy), u2(ix,iy-1), u2(ix,iy), u2(ix,iy+1) );
            limdiff( gy2(ix,iy), g2(ix,iy-1), g2(ix,iy), g2(ix,iy+1) );

            // // x derivs
            // limdiff( ux(ix,iy), u(ix-1,iy), u(ix,iy), u(ix+1,iy) );
            // limdiff( fx(ix,iy), f(ix-1,iy), f(ix,iy), f(ix+1,iy) );

            // // y derivs
            // limdiff( uy(ix,iy), u(ix,iy-1), u(ix,iy), u(ix,iy+1) );
            // limdiff( gy(ix,iy), g(ix,iy-1), g(ix,iy), g(ix,iy+1) );
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

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::compute_step(int io, real dt)
{
    real dtcdx2 = 0.5 * dt / dx;
    real dtcdy2 = 0.5 * dt / dy;

    // Predictor (flux values of f and g at half step)
    for (int iy = 1; iy < ny_all-1; ++iy) {
        #pragma ivdep
        for (int ix = 1; ix < nx_all-1; ++ix) {
            // vec uh = u(ix,iy);
            // for (int m = 0; m < uh.size(); ++m) {
            //     uh[m] -= dtcdx2 * fx(ix,iy)[m];
            //     uh[m] -= dtcdy2 * gy(ix,iy)[m];
            // }
            // Physics::flux(f(ix,iy), g(ix,iy), uh);
            real ut0 = u0(ix,iy);
            real ut1 = u1(ix,iy);
            real ut2 = u2(ix,iy);
            ut0 -= dtcdx2 * fx0(ix,iy);
            ut0 -= dtcdy2 * gy0(ix,iy);
            ut1 -= dtcdx2 * fx1(ix,iy);
            ut1 -= dtcdy2 * gy1(ix,iy);
            ut2 -= dtcdx2 * fx2(ix,iy);
            ut2 -= dtcdy2 * gy2(ix,iy);
            Physics::flux(f0(ix,iy), g0(ix,iy), ut0,
                          f1(ix,iy), g1(ix,iy), ut1,
                          f2(ix,iy), g2(ix,iy), ut2);
        }
    }

    const int outer = ny+nghost-io;
    const int inner = nx+nghost-io;
    // Corrector (finish the step)
    for (int iy = nghost-io; iy < outer; ++iy) {
        #pragma ivdep
        for (int ix = nghost-io; ix < inner; ++ix) {
            v0(ix,iy) =
                0.2500 * ( u0(ix,  iy) + u0(ix+1,iy  ) +
                           u0(ix,iy+1) + u0(ix+1,iy+1) ) -
                0.0625 * ( ux0(ix+1,iy  ) - ux0(ix,iy  ) +
                           ux0(ix+1,iy+1) - ux0(ix,iy+1) +
                           uy0(ix,  iy+1) - uy0(ix,  iy) +
                           uy0(ix+1,iy+1) - uy0(ix+1,iy) ) -
                dtcdx2 * ( f0(ix+1,iy  ) - f0(ix,iy  ) +
                           f0(ix+1,iy+1) - f0(ix,iy+1) ) -
                dtcdy2 * ( g0(ix,  iy+1) - g0(ix,  iy) +
                           g0(ix+1,iy+1) - g0(ix+1,iy) );
        }
    }

    for (int iy = nghost-io; iy < outer; ++iy) {
        #pragma ivdep
        for (int ix = nghost-io; ix < inner; ++ix) {
            v1(ix,iy) =
                0.2500 * ( u1(ix,  iy) + u1(ix+1,iy  ) +
                           u1(ix,iy+1) + u1(ix+1,iy+1) ) -
                0.0625 * ( ux1(ix+1,iy  ) - ux1(ix,iy  ) +
                           ux1(ix+1,iy+1) - ux1(ix,iy+1) +
                           uy1(ix,  iy+1) - uy1(ix,  iy) +
                           uy1(ix+1,iy+1) - uy1(ix+1,iy) ) -
                dtcdx2 * ( f1(ix+1,iy  ) - f1(ix,iy  ) +
                           f1(ix+1,iy+1) - f1(ix,iy+1) ) -
                dtcdy2 * ( g1(ix,  iy+1) - g1(ix,  iy) +
                           g1(ix+1,iy+1) - g1(ix+1,iy) );
        }
    }

    for (int iy = nghost-io; iy < outer; ++iy) {
        #pragma ivdep
        for (int ix = nghost-io; ix < inner; ++ix) {
            v2(ix,iy) =
                0.2500 * ( u2(ix,  iy) + u2(ix+1,iy  ) +
                           u2(ix,iy+1) + u2(ix+1,iy+1) ) -
                0.0625 * ( ux2(ix+1,iy  ) - ux2(ix,iy  ) +
                           ux2(ix+1,iy+1) - ux2(ix,iy+1) +
                           uy2(ix,  iy+1) - uy2(ix,  iy) +
                           uy2(ix+1,iy+1) - uy2(ix+1,iy) ) -
                dtcdx2 * ( f2(ix+1,iy  ) - f2(ix,iy  ) +
                           f2(ix+1,iy+1) - f2(ix,iy+1) ) -
                dtcdy2 * ( g2(ix,  iy+1) - g2(ix,  iy) +
                           g2(ix+1,iy+1) - g2(ix+1,iy) );
        }
    }

    // Copy from v storage back to main grid
    for (int j = nghost; j < ny+nghost; ++j) {
        for (int i = nghost; i < nx+nghost; ++i) {
            // u(i,j) = v(i-io,j-io);
            u0(i,j) = v0(i-io,j-io);
            u1(i,j) = v1(i-io,j-io);
            u2(i,j) = v2(i-io,j-io);
        }
    }
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

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::run(real tfinal)
{
    bool done = false;
    real t = 0;
    while (!done) {
        real dt;
        for (int io = 0; io < 2; ++io) {
            real cx, cy;
            apply_periodic();
            compute_fg_speeds(cx, cy);
            limited_derivs();
            if (io == 0) {
                dt = cfl / std::max(cx/dx, cy/dy);
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

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::solution_check()
{
    using namespace std;
    real h_sum = 0, hu_sum = 0, hv_sum = 0;
    real hmin = u0(nghost,nghost);
    real hmax = hmin;
    for (int j = nghost; j < ny+nghost; ++j)
        for (int i = nghost; i < nx+nghost; ++i) {
            // vec& uij = u(i,j);
            real h = u0(i,j);
            h_sum += h;
            hu_sum += u1(i,j);
            hv_sum += u2(i,j);
            hmax = max(h, hmax);
            hmin = min(h, hmin);
            assert( h > 0) ;
        }
    real cell_area = dx*dy;
    h_sum *= cell_area;
    hu_sum *= cell_area;
    hv_sum *= cell_area;
    printf("-\n  Volume: %g\n  Momentum: (%g, %g)\n  Range: [%g, %g]\n",
           h_sum, hu_sum, hv_sum, hmin, hmax);
}

//ldoc off
#endif /* CENTRAL2D_H*/
