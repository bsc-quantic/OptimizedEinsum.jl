ENV["PYTHON"] = ""

using Pkg
Pkg.build("PyCall")

using Conda
Conda.add("opt_einsum")