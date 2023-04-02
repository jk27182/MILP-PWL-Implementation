file=$1
file_name=$2
clang++ $file -o $file_name -I/Applications/CPLEX_Studio2211/cplex/include -I/Applications/CPLEX_Studio2211/concert/include  -DIL_STD -L/Applications/CPLEX_Studio2211/cplex/lib/arm64_osx/static_pic  -L/Applications/CPLEX_Studio2211/concert/lib/arm64_osx/static_pic -lilocplex -lconcert -lcplex -lm -lpthread

