import argparse, os, json
import numpy as np
from scipy.integrate import solve_ivp
import pysindy as ps
# import all your ODE functions
from pysindy.utils import (
    linear_damped_SHO, cubic_damped_SHO, linear_3D, hopf, lorenz
)

def run_sindy(ode_fn, poly_order, threshold, dt, t_train, x0):
    # 1. simulate
    sol = solve_ivp(ode_fn, (t_train[0], t_train[-1]), x0,
                    t_eval=t_train, rtol=1e-12, atol=1e-12, method='LSODA')
    x_train = sol.y.T

    # 2. fit
    model = ps.SINDy(
      optimizer=ps.STLSQ(threshold=threshold),
      feature_library=ps.PolynomialLibrary(degree=poly_order),
    )
    model.fit(x_train, t=dt)

    # 3. simulate back
    x_sim = model.simulate(x0, t_train)

    np.save(os.path.join(args.out_dir, "t_train.npy"), t_train)
    np.save(os.path.join(args.out_dir, "x_train.npy"), x_train)
    np.save(os.path.join(args.out_dir, "x_sim.npy"),   x_sim)

    # 2) Save coefficients
    coef_dict = {
        term: float(val)
        for term, val in zip(model.get_feature_names(),
                             model.coefficients().ravel())
    }
    with open(os.path.join(args.out_dir, "coefficients.json"), "w") as f:
        json.dump(coef_dict, f, indent=2)
        
    # 4. compute a scalar metric, e.g. RMSE
    rmse = np.sqrt(np.mean((x_train - x_sim)**2))
    return rmse, model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="run_0")
    parser.add_argument("--ode_fn", type=str, default="linear_damped_SHO")
    parser.add_argument("--poly_order", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--t_end", type=float, default=25.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--x0", nargs="+", type=float, default=[2, 0])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # look up the function by name
    ode_fn = globals()[args.ode_fn]

    # build the time vector
    t_train = np.arange(0, args.t_end, args.dt)

    rmse, model = run_sindy(
        ode_fn, args.poly_order, args.threshold,
        args.dt, t_train, args.x0
    )

    # write out the JSON in the required structure
    final_info = {
        "rmse": {
            "means": float(rmse),
            "stderrs": 0.0
        }
    }
    with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
        json.dump(final_info, f, indent=2)
