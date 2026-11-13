	Julia Installation
curl -fsSL https://install.julialang.org | sh



	Julia (Comet rotation Package Instantiate)
julia --project -e 'using Pkg; Pkg.instantiate()'

	Running the example scripts 
julia --project -t auto forward_integration.jl
julia --project -t auto fit_local.jl
julia --project -t auto fit_global.jl 2 16

	Using REPL to run the scripts
Run this in Julia Global environment : julia -e 'using Pkg; Pkg.add(["Revise", "IJulia"])'
cd example directory and run : julia --project -t auto

	Inside REPL
using Revise
Include("script_name.jl)

	Using Jupyter Lab to run the scripts
Inside REPL(One time) : using IJulia
			IJulia.installkernel("Julia (auto threads)", "--project=@.";
                     				env = Dict("JULIA_NUM_THREADS" => "auto"))

julia -e 'using IJulia; jupyterlab(dir=pwd())'   (This opens Jupyter lab in local host)

	In the first cell: 

   using Pkg
   Pkg.activate("path_to_CometRotation.jl")
   using Revise
   using CometRotation
   
	In the 2nd cell :
  
  cd("path to script 1 or 2")
  include("script_name.jl")



