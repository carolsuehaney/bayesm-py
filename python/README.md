
## Structure

- `bayesm/` - Main package
  - `constants.py` - MCMC and prior constants
  - `utilities.py` - Core utility functions
  - `create_x.py` - Design matrix creation
  - `_cpp/` - Compiled C++ extensions
- `tests/` - Test suite
- `cpp_bindings/` - C++ source and build files

## Status

This is an active conversion from R to Python. See `../CONVERSION_PLAN.md` for details.

## Core Utilities Converted

- [ ] constants (BayesmConstants.R)
- [ ] utilities (BayesmFunctions.R, nmat.R)
- [ ] create_x (createX.R)
- [ ] cond_mom (condMom.R)
- [ ] num_eff (numEff.R)

## C++ Utilities

Status: Not yet compiled

## Testing

```bash
pytest tests/
```

## License

GPL (>= 2) - matching original R package
