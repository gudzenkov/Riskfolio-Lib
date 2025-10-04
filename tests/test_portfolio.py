""""""  #
"""
Copyright (c) 2020-2022, Dany Cajas
All rights reserved.
This work is licensed under BSD 3-Clause "New" or "Revised" License.
License available at https://github.com/dcajasn/Riskfolio-Lib/blob/master/LICENSE.txt
"""

import os
import numpy as np
import pandas as pd
import riskfolio as rp

assets = ["JCI", "TGT", "CMCSA", "CPB", "MO", "AMZN", "APA", "MMC", "JPM", "ZION"]
assets.sort()
benchmark = ["SPY"]


def resource(name):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), name)


def get_data(name):
    return pd.read_csv(resource(name), parse_dates=True, index_col=0)


def test_classic_minrisk_optimization():

    Y = get_data("stock_prices.csv")
    Y = Y[assets].pct_change().dropna().iloc[-200:]

    port = rp.Portfolio(returns=Y)

    method_mu = "hist"
    method_cov = "hist"

    port.assets_stats(method_mu=method_mu, method_cov=method_cov)
    port.alpha = 0.05
    port.solvers = ['CLARABEL', 'SCS', 'ECOS']

    model = "Classic"
    obj = "MinRisk"
    hist = True
    rf = 0
    l = 0

    rms = [
        "MV",
        "MAD",
        "GMD",
        "MSV",
        "FLPM",
        "SLPM",
        "CVaR",
        "TG",
        "EVaR",
        "WR",
        "RG",
        "CVRG",
        "TGRG",
        "MDD",
        "ADD",
        "CDaR",
        "EDaR",
        "UCI",
    ]

    w_1 = pd.DataFrame([])

    for i in rms:
        w = port.optimization(model=model, rm=i, obj=obj, rf=rf, l=l, hist=hist)
        w_1 = pd.concat([w_1, w], axis=1)

    w_1.columns = rms
    # w_1.to_csv(resource("Classic_MinRisk.csv"))

    w_2 = get_data("Classic_MinRisk.csv")

    a = np.testing.assert_array_almost_equal(w_1.to_numpy(), w_2.to_numpy(), decimal=6)
    if a is None:
        print("There are no errors in test_classic_minrisk_optimization")


def test_classic_sharpe_optimization():

    Y = get_data("stock_prices.csv")
    Y = Y[assets].pct_change().dropna().iloc[-200:]

    port = rp.Portfolio(returns=Y)

    method_mu = "hist"
    method_cov = "hist"

    port.assets_stats(method_mu=method_mu, method_cov=method_cov)
    port.alpha = 0.05
    port.solvers = ['CLARABEL', 'SCS', 'ECOS']

    model = "Classic"
    obj = "Sharpe"
    hist = True
    rf = 0
    l = 0

    rms = [
        "MV",
        "MAD",
        "GMD",
        "MSV",
        "FLPM",
        "SLPM",
        "CVaR",
        "TG",
        "EVaR",
        "WR",
        "RG",
        "CVRG",
        "TGRG",
        "MDD",
        "ADD",
        "CDaR",
        "EDaR",
        "UCI",
    ]

    w_1 = pd.DataFrame([])

    for i in rms:
        w = port.optimization(model=model, rm=i, obj=obj, rf=rf, l=l, hist=hist)
        w_1 = pd.concat([w_1, w], axis=1)

    w_1.columns = rms
    # w_1.to_csv(resource("Classic_Sharpe.csv"))

    w_2 = get_data("Classic_Sharpe.csv")

    a = np.testing.assert_array_almost_equal(w_1.to_numpy(), w_2.to_numpy(), decimal=6)
    if a is None:
        print("There are no errors in test_classic_sharpe_optimization")


def test_classic_riskparity_optimization():

    Y = get_data("stock_prices.csv")
    Y = Y[assets].pct_change().dropna().iloc[-200:]

    port = rp.Portfolio(returns=Y)

    method_mu = "hist"
    method_cov = "hist"

    port.assets_stats(method_mu=method_mu, method_cov=method_cov)
    port.alpha = 0.05
    port.solvers = ['CLARABEL', 'ECOS', 'SCS']

    model = "Classic"
    hist = True
    rf = 0
    b = None

    rms = [
        "MV",
        "MAD",
        "GMD",
        "MSV",
        "FLPM",
        "SLPM",
        "CVaR",
        "TG",
        "CVRG",
        "TGRG",
        "EVaR",
        "CDaR",
        "EDaR",
        "UCI",
    ]

    w_1 = pd.DataFrame([])

    for i in rms:
        w = port.rp_optimization(model=model, rm=i, rf=rf, b=b, hist=hist)
        w_1 = pd.concat([w_1, w], axis=1)

    w_1.columns = rms
    # w_1.to_csv(resource("Classic_RP.csv"))

    w_2 = get_data("Classic_RP.csv")

    a = np.testing.assert_array_almost_equal(w_1.to_numpy(), w_2.to_numpy(), decimal=6)
    if a is None:
        print("There are no errors in test_classic_riskparity_optimization")


def test_worst_case_optimization():

    Y = get_data("stock_prices.csv")
    Y = Y[assets].pct_change().dropna().iloc[-200:]

    port = rp.Portfolio(returns=Y)

    method_mu = "hist"
    method_cov = "hist"

    port.assets_stats(method_mu=method_mu, method_cov=method_cov)
    port.solvers = ['CLARABEL', 'ECOS', 'SCS']

    box = 's'
    ellip = 's'
    q = 0.05
    n_sim = 3000
    window = 3
    dmu = 0.1
    dcov = 0.1
    seed = 0

    port.wc_stats(box=box, ellip=ellip, q=q, n_sim=n_sim, window=window, dmu=dmu, dcov=dcov, seed=seed)

    Umus = ['box', 'ellip']
    Ucovs = ['box', 'ellip']
    objs = ['MinRisk', 'Sharpe']
    rf = 0
    l = 0

    w_1 = pd.DataFrame([])
    headers = []
    for obj in objs:
        for Umu in Umus:
            for Ucov in Ucovs:
                w = port.wc_optimization(obj=obj, rf=rf, l=l, Umu=Umu, Ucov=Ucov)
                w_1 = pd.concat([w_1, w], axis=1)
                headers += [obj + '-' + Umu + '-' + Ucov]

    w_1.columns = headers
    # w_1.to_csv(resource("Classic_WC.csv"))

    w_2 = get_data("Classic_WC.csv")

    a = np.testing.assert_array_almost_equal(w_1.to_numpy(), w_2.to_numpy(), decimal=6)
    if a is None:
        print("There are no errors in test_worst_case_optimization")


def test_hc_hrp_optimization():

    Y = get_data("stock_prices.csv")
    Y = Y[assets].pct_change().dropna().iloc[-200:]

    port = rp.HCPortfolio(returns=Y)

    model = "HRP"
    codependence = "pearson"
    rf = 0
    linkage = "single"
    max_k = 10
    leaf_order = True

    rms = [
        "vol",
        "MV",
        "MAD",
        "GMD",
        "MSV",
        "FLPM",
        "SLPM",
        "VaR",
        "CVaR",
        "TG",
        "EVaR",
        "WR",
        "RG",
        "CVRG",
        "TGRG",
        "MDD",
        "ADD",
        "DaR",
        "CDaR",
        "EDaR",
        "UCI",
        "MDD_Rel",
        "ADD_Rel",
        "DaR_Rel",
        "CDaR_Rel",
        "EDaR_Rel",
        "UCI_Rel",
    ]

    w_1 = pd.DataFrame([])

    for i in rms:
        w = port.optimization(
            model=model,
            codependence=codependence,
            rm=i,
            rf=rf,
            linkage=linkage,
            max_k=max_k,
            leaf_order=leaf_order,
        )

        w_1 = pd.concat([w_1, w], axis=1)

    w_1.columns = rms
    # w_1.to_csv(resource("HC_HRP.csv"))

    w_2 = get_data("HC_HRP.csv")

    a = np.testing.assert_array_almost_equal(w_1.to_numpy(), w_2.to_numpy(), decimal=6)
    if a is None:
        print("There are no errors in test_hc_hrp_optimization")


def test_hc_herc_optimization():

    Y = get_data("stock_prices.csv")
    Y = Y[assets].pct_change().dropna().iloc[-200:]

    port = rp.HCPortfolio(returns=Y)

    model = "HERC"
    codependence = "pearson"
    rf = 0
    linkage = "ward"
    max_k = 10
    leaf_order = True

    rms = [
        "vol",
        "MV",
        "MAD",
        "GMD",
        "MSV",
        "FLPM",
        "SLPM",
        "VaR",
        "CVaR",
        "TG",
        "EVaR",
        "WR",
        "RG",
        "CVRG",
        "TGRG",
        "MDD",
        "ADD",
        "DaR",
        "CDaR",
        "EDaR",
        "UCI",
        "MDD_Rel",
        "ADD_Rel",
        "DaR_Rel",
        "CDaR_Rel",
        "EDaR_Rel",
        "UCI_Rel",
    ]

    w_1 = pd.DataFrame([])

    for i in rms:
        w = port.optimization(
            model=model,
            codependence=codependence,
            rm=i,
            rf=rf,
            linkage=linkage,
            max_k=max_k,
            leaf_order=leaf_order,
        )

        w_1 = pd.concat([w_1, w], axis=1)

    w_1.columns = rms
    # w_1.to_csv(resource("HC_HERC.csv"))

    w_2 = get_data("HC_HERC.csv")

    a = np.testing.assert_array_almost_equal(w_1.to_numpy(), w_2.to_numpy(), decimal=6)
    if a is None:
        print("There are no errors in test_hc_herc_optimization")


def test_hc_nco_optimization():

    Y = get_data("stock_prices.csv")
    Y = Y[assets].pct_change().dropna().iloc[-200:]

    port = rp.HCPortfolio(returns=Y)

    model = "NCO"
    codependence = "pearson"
    method_cov = "hist"
    obj = "MinRisk"
    rf = 0
    linkage = "ward"
    max_k = 10
    leaf_order = True

    rms = [
        "MV",
        "MAD",
        "MSV",
        "FLPM",
        "SLPM",
        "CVaR",
        "EVaR",
        "WR",
        "MDD",
        "ADD",
        "CDaR",
        "EDaR",
        "UCI",
    ]

    w_1 = pd.DataFrame([])

    for i in rms:
        w = port.optimization(
            model=model,
            codependence=codependence,
            method_cov=method_cov,
            obj=obj,
            rm=i,
            rf=rf,
            linkage=linkage,
            max_k=max_k,
            leaf_order=leaf_order,
        )

        w_1 = pd.concat([w_1, w], axis=1)

    w_1.columns = rms
    # w_1.to_csv(resource("HC_NCO.csv"))

    w_2 = get_data("HC_NCO.csv")

    a = np.testing.assert_array_almost_equal(w_1.to_numpy(), w_2.to_numpy(), decimal=6)
    if a is None:
        print("There are no errors in test_hc_nco_optimization")


def test_budgetcap_optimization():
    """Test budgetcap (gross exposure) constraint with long-short portfolios."""

    Y = get_data("stock_prices.csv")
    Y = Y[assets].pct_change().dropna().iloc[-200:]

    port = rp.Portfolio(returns=Y)

    method_mu = "hist"
    method_cov = "hist"

    port.assets_stats(method_mu=method_mu, method_cov=method_cov)
    port.alpha = 0.05
    port.solvers = ['CLARABEL', 'SCS', 'ECOS']

    model = "Classic"
    hist = True
    rf = 0
    l = 0

    # Test 1: Scenario 1 - Only N is set (budgetcap=None) - Default behavior
    print("Testing Scenario 1: Only N is set (default behavior)")
    port.sht = True
    port.budget = 1.0
    port.budgetcap = None

    w_scenario1 = port.optimization(model=model, rm="MV", obj="Sharpe", rf=rf, l=l, hist=hist)

    gross1 = np.abs(w_scenario1.to_numpy()).sum()
    net1 = w_scenario1.to_numpy().sum()

    # Verify net constraint
    assert abs(net1 - port.budget) < 1e-6, f"Scenario 1: Net constraint violated: {net1} != {port.budget}"
    print(f"  ✓ Scenario 1: Net={net1:.6f}, Gross={gross1:.6f} (default behavior preserved)")

    # Test 2: Scenario 2 - Gross set with feasible net
    # Note: budgetcap must be feasible for the data. If budgetcap > natural optimum,
    # constraint may not be satisfied. Use budgetcap <= natural gross for reliability.
    print("Testing Scenario 2: Only G is set (N unconstrained)")
    port.sht = True
    port.budget = None  # N is unconstrained
    port.budgetcap = 1.2  # G is set

    # Feasibility check: Verify budgetcap is achievable
    # First optimize without budgetcap to find natural gross
    port_temp = rp.Portfolio(returns=Y)
    port_temp.assets_stats(method_mu=method_mu, method_cov=method_cov)
    port_temp.sht = True
    w_free = port_temp.optimization(model=model, rm="MV", obj="Utility", rf=rf, l=2, hist=True)
    natural_gross = np.abs(w_free.to_numpy()).sum()

    if port.budgetcap > natural_gross * 1.05:  # 5% tolerance
        print(f"  ⚠ Warning: budgetcap {port.budgetcap:.3f} > natural gross {natural_gross:.3f} - constraint may be slack")

    # Use Utility objective (Sharpe doesn't support budget=None with budgetcap)
    # l=2 is used to balance risk and return while allowing unconstrained net
    w_scenario2 = port.optimization(model=model, rm="MV", obj="Utility", rf=rf, l=2, hist=True)

    gross2 = np.abs(w_scenario2.to_numpy()).sum()
    net2 = w_scenario2.to_numpy().sum()

    gross_error2 = abs(gross2 - port.budgetcap)

    # Critical: Verify complementarity (wp[i] * wn[i] ≈ 0)
    # This ensures sum(|w|) = sum(wp + wn)
    w_opt2 = w_scenario2.to_numpy().flatten()
    wp2 = np.maximum(w_opt2, 0)
    wn2 = np.maximum(-w_opt2, 0)
    overlap2 = wp2 * wn2

    assert np.all(overlap2 < 1e-8), f"Scenario 2: Complementarity violated: max overlap = {overlap2.max()}"
    assert gross_error2 < 1e-6, f"Scenario 2: Gross constraint violated: {gross2} != {port.budgetcap}"

    # Validate net exposure is within gross bounds
    assert abs(net2) <= gross2 + 1e-6, f"Scenario 2: Net {net2} exceeds gross {gross2}"

    print(f"  ✓ Scenario 2: Gross={gross2:.6f} (error: {gross_error2:.2e}), Net={net2:.6f} (unconstrained)")
    print(f"             Complementarity: max(wp*wn) = {overlap2.max():.2e}, Net/Gross = {abs(net2)/gross2:.1%}")

    # Test 3: Scenario 3 (Optional) - Both G & N are set - Both equality constraints followed
    print("Testing Scenario 3: Both G & N are set (optional)")
    port.sht = True
    port.budget = 1.0
    port.budgetcap = 1.6

    w_scenario3_sharpe = port.optimization(model=model, rm="MV", obj="Sharpe", rf=rf, l=l, hist=hist)

    gross3 = np.abs(w_scenario3_sharpe.to_numpy()).sum()
    net3 = w_scenario3_sharpe.to_numpy().sum()

    # Feasibility checks
    gross_error = abs(gross3 - port.budgetcap)
    net_error = abs(net3 - port.budget)

    assert gross_error < 1e-6, f"Scenario 3 Sharpe: Gross constraint violated: {gross3} != {port.budgetcap}"
    assert net_error < 1e-6, f"Scenario 3 Sharpe: Net constraint violated: {net3} != {port.budget}"

    # Complementarity check
    w_opt = w_scenario3_sharpe.to_numpy().flatten()
    w_long_check = np.maximum(w_opt, 0)
    w_short_check = np.maximum(-w_opt, 0)
    overlap = w_long_check * w_short_check

    assert np.all(overlap < 1e-8), f"Scenario 3 Sharpe: Complementarity violated: max overlap = {overlap.max()}"

    print(f"  ✓ Scenario 3 Sharpe: Gross={gross3:.6f} (error: {gross_error:.2e}), Net={net3:.6f} (error: {net_error:.2e})")

    # Test 3: Different budgetcap values
    print("Testing with budgetcap = 2.0")
    port.budgetcap = 2.0

    w_budgetcap2 = port.optimization(model=model, rm="MV", obj="Sharpe", rf=rf, l=l, hist=hist)

    gross4 = np.abs(w_budgetcap2.to_numpy()).sum()
    net4 = w_budgetcap2.to_numpy().sum()

    gross_error4 = abs(gross4 - port.budgetcap)
    net_error4 = abs(net4 - port.budget)

    assert gross_error4 < 1e-6, f"budgetcap=2.0: Gross constraint violated: {gross4} != {port.budgetcap}"
    assert net_error4 < 1e-6, f"budgetcap=2.0: Net constraint violated: {net4} != {port.budget}"

    print(f"  ✓ budgetcap=2.0: Gross={gross4:.6f} (error: {gross_error4:.2e}), Net={net4:.6f} (error: {net_error4:.2e})")

    print("There are no errors in test_budgetcap_optimization")


if __name__ == '__main__':
    test_classic_minrisk_optimization()
    test_classic_sharpe_optimization()
    test_classic_riskparity_optimization()
    test_worst_case_optimization()
    test_hc_hrp_optimization()
    test_hc_herc_optimization()
    test_hc_nco_optimization()
    test_budgetcap_optimization()