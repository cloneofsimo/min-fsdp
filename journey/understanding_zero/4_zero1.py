# So why is 3 not Zero-1 yet?
# 1. We need to do mixed-precision!
# 2. gradients could be in better form, we can use hooks to form gradients in unfragmented way.
# 3. minor details, such as skipping small params / checking for required_grad, and accepting param_group as input is all missing.
