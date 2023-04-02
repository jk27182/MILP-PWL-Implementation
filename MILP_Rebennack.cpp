// -------------------------------------------------------------- -*- C++ -*-

#include <ilcplex/ilocplex.h>
// /Applications/CPLEX_Studio221/cplex/include
// /Applications/CPLEX_Studio221/concert/include
// 
//
//
// clang++ -I/Applications/CPLEX_Studio2211/cplex/include -I/Applications/CPLEX_Studio2211/concert/include  -DIL_STD -L/Applications/CPLEX_Studio2211/cplex/lib/arm64_osx/static_pic  -L/Applications/CPLEX_Studio2211/concert/lib/arm64_osx/static_pic -lilocplex -lconcert -lcplex -lm -lpthread
ILOSTLBEGIN
using namespace std;

typedef IloArray<IloNumVarArray> NumVarMatrix;
typedef IloArray<IloNumArray>    NumMatrix;
typedef IloArray<IloBoolVarArray> BoolVarMatrix;

std::string filename = "data/MpStorage50.txt";
const IloInt LinSeg = 3;
const IloInt Breakpoints = LinSeg + 1;
IloInt Objective = 2; //Pick Distance Metric. 0=Feasibility, 1=LInf, 2=L1, 3=L2

int main(){
	int i, j;
	ifstream inFile;
	inFile.open(filename);
	if (!inFile) {
		cout << "Unable to open file";
		exit(1); // terminate with error
	}
	int Filesize = 0;
	double count[1];
	while (inFile.good()) {
		inFile >> count[0];
		Filesize += 1;
	}
	const IloInt DataPoints = Filesize / 2;
	std::cout << "Number of Datapoints: " << DataPoints << std::endl;
	inFile.close();
	double** data = new double* [DataPoints];
	for (i = 0; i < DataPoints; ++i)
	{
		data[i] = new double[2];
	}
	inFile.open(filename);
	while (inFile.good()) {
		for (i = 0; i < DataPoints; i++) {
			for (j = 0; j < 2; j++) {
				inFile >> data[i][j];
			}
		}
	}

	double cmax = -IloInfinity;
	double cmin = IloInfinity;
	// For the first iteration the second loop is not accessed, is not a problem though
	for (i = 0; i < DataPoints; i++) {
		for (j = 0; j < i; j++) {
			if ((data[i][1] - data[j][1]) / (data[i][0] - data[j][0]) > cmax) {
				cmax = (data[i][1] - data[j][1]) / (data[i][0] - data[j][0]);
			}
			if ((data[i][1] - data[j][1]) / (data[i][0] - data[j][0]) < cmin) {
				cmin = (data[i][1] - data[j][1]) / (data[i][0] - data[j][0]);
			}
		}
	}

	double dmax = -IloInfinity;
	double dmin = IloInfinity;
	for (i = 0; i < DataPoints; i++) {
		if (data[i][1] - cmax * data[i][0] > dmax) {
			dmax = data[i][1] - cmax * data[i][0];
		}
		if (data[i][1] - cmin * data[i][0] > dmax) {
			dmax = data[i][1] - cmin * data[i][0];
		}
		if (data[i][1] - cmax * data[i][0] < dmin) {
			dmin = data[i][1] - cmax * data[i][0];
		}
		if (data[i][1] - cmin * data[i][0] < dmin) {
			dmin = data[i][1] - cmin * data[i][0];
		}
	}

	double* M_a = new double[DataPoints];
	double* M_2 = new double[DataPoints];
	for (i = 0; i < DataPoints; i++) {
		M_a[i] = data[i][1] - cmax * data[i][0] - dmax;
		if (data[i][1] - cmin * data[i][0] - dmax > M_a[i]) {
			M_a[i] = data[i][1] - cmin * data[i][0] - dmax;
		}
		if (data[i][1] - cmax * data[i][0] - dmin > M_a[i]) {
			M_a[i] = data[i][1] - cmax * data[i][0] - dmin;
		}
		if (data[i][1] - cmin * data[i][0] - dmin > M_a[i]) {
			M_a[i] = data[i][1] - cmin * data[i][0] - dmin;
		}
		M_2[i] = dmax - dmin - data[i][0] * (cmin - cmax);
	}
	IloEnv env;
	try {
		//------------------------------------------------------Variable Definitions------------------------------------------
		IloNumVarArray c(env, Breakpoints - 1, cmin, cmax);
		IloNumVarArray d(env, Breakpoints - 1, dmin, dmax);
		IloBoolVarArray gamma(env, Breakpoints - 2);
		BoolVarMatrix delta(env, DataPoints);
		NumVarMatrix deltaplus(env, DataPoints - 1);
		NumVarMatrix deltaminus(env, DataPoints - 1);
		IloNumVarArray epsilon(env, DataPoints, 0, IloInfinity);
		IloInt i, b;

		for (i = 0; i < DataPoints; i++) {
			delta[i] = IloBoolVarArray(env, Breakpoints - 1);
		}
		for (i = 0; i < DataPoints - 1; i++) {
			deltaplus[i] = IloNumVarArray(env, Breakpoints - 2, 0, 1);
			deltaminus[i] = IloNumVarArray(env, Breakpoints - 2, 0, 1);
		}

		//-------------------------------------------------------Adding Constraints-------------------------------------------------------------
		IloModel model(env);

		for (i = 0; i < DataPoints; i++) {
			IloNumExpr sum_over_i(env);
			for (b = 0; b < Breakpoints - 1; b++) {
				sum_over_i += delta[i][b];
			}
			model.add(sum_over_i == 1);
		}

		for (i = 0; i < DataPoints - 1; i++) {
			for (b = 0; b < Breakpoints - 2; b++) {
				model.add(delta[i + 1][b + 1] <= delta[i][b] + delta[i][b + 1]);
			}
			model.add(delta[i + 1][0] <= delta[i][0]);
			model.add(delta[i][Breakpoints - 2] <= delta[i + 1][Breakpoints - 2]);
		}

		for (i = 0; i < DataPoints - 1; i++) {
			for (b = 0; b < Breakpoints - 2; b++) {
				model.add(delta[i][b] + delta[i + 1][b + 1] + gamma[b] - 2 <= deltaplus[i][b]);
				model.add(delta[i][b] + delta[i + 1][b + 1] + (1 - gamma[b]) - 2 <= deltaminus[i][b]);
				model.add(d[b + 1] - d[b] >= data[i][0] * (c[b] - c[b + 1]) - M_2[i] * (1 - deltaplus[i][b]));
				model.add(d[b + 1] - d[b] <= data[i + 1][0] * (c[b] - c[b + 1]) + M_2[i + 1] * (1 - deltaplus[i][b]));
				model.add(d[b + 1] - d[b] <= data[i][0] * (c[b] - c[b + 1]) + M_2[i] * (1 - deltaminus[i][b]));
				model.add(d[b + 1] - d[b] >= data[i + 1][0] * (c[b] - c[b + 1]) - M_2[i + 1] * (1 - deltaminus[i][b]));
			}
		}

		//-------------------------------------------------Model Objective--------------------------------------

		if (Objective == 0) {
			model.add(IloMinimize(env, 1));
		}

		else if (Objective == 1) {
			IloNumVar tau(env, 0, IloInfinity);
			for (i = 0; i < DataPoints; i++) {
				for (b = 0; b < Breakpoints - 1; b++) {
					model.add(data[i][1] - (c[b] * data[i][0] + d[b]) <= tau + M_a[i] * (1 - delta[i][b]));
					model.add((c[b] * data[i][0] + d[b]) - data[i][1] <= tau + M_a[i] * (1 - delta[i][b]));
				}
			}
			model.add(IloMinimize(env, tau));
		}

		else if (Objective == 2) {
			for (i = 0; i < DataPoints; i++) {
				for (b = 0; b < Breakpoints - 1; b++) {
					model.add(data[i][1] - (c[b] * data[i][0] + d[b]) <= epsilon[i] + M_a[i] * (1 - delta[i][b]));
					model.add((c[b] * data[i][0] + d[b]) - data[i][1] <= epsilon[i] + M_a[i] * (1 - delta[i][b]));
				}
			}
			IloNumExpr objective1(env);
			for (i = 0; i < DataPoints; i++) {
				objective1 += epsilon[i];
			}
			model.add(IloMinimize(env, objective1));
		}

		else if (Objective == 3) {
			for (i = 0; i < DataPoints; i++) {
				for (b = 0; b < Breakpoints - 1; b++) {
					model.add(data[i][1] - (c[b] * data[i][0] + d[b]) <= epsilon[i] + M_a[i] * (1 - delta[i][b]));
					model.add((c[b] * data[i][0] + d[b]) - data[i][1] <= epsilon[i] + M_a[i] * (1 - delta[i][b]));
				}
			}
			IloNumExpr objective2(env);
			for (i = 0; i < DataPoints; i++) {
				objective2 += epsilon[i] * epsilon[i];
			}
			model.add(IloMinimize(env, objective2));
		}

		//-------------------------------------------------Solve the Model -------------------------------------------------

		IloCplex cplex(model);
		cplex.solve();

		cout << endl << "Solution status: " << cplex.getStatus() << endl;
		cout << "Objective Value = " << cplex.getObjValue() << endl << endl;

		IloBool WriteToFile = IloFalse;
		if (WriteToFile) {
			ofstream solfile;
			solfile.open("Functions.txt");
			solfile << "Value = " << cplex.getObjValue() << endl << "Affine Functions: " << endl;
			solfile << cplex.getValue(c[0]) << " * X + " << cplex.getValue(d[0]) << " (" << data[0][0] << " <= X <= " << cplex.getValue((d[1] - d[0]) / (c[0] - c[1])) << ")" << endl;
			for (b = 1; b < Breakpoints - 2; b++) {
				solfile << cplex.getValue(c[b]) << " * X + " << cplex.getValue(d[b]) << " (" << cplex.getValue((d[b] - d[b - 1]) / (c[b - 1] - c[b])) << " <= X <= " << cplex.getValue((d[b + 1] - d[b]) / (c[b] - c[b + 1])) << ")" << endl;
			}
			solfile << cplex.getValue(c[Breakpoints - 2]) << " * X + " << cplex.getValue(d[Breakpoints - 2]) << " (" << cplex.getValue((d[Breakpoints - 2] - d[Breakpoints - 3]) / (c[Breakpoints - 3] - c[Breakpoints - 2])) << " <= X <= " << data[DataPoints - 1][0] << ")" << endl;
			solfile << endl;
			solfile.close();
		}

		// IloBool Print = IloFalse;
		bool Print = true;
		if (Print) {
			ofstream res_file;
			res_file.open("res_rebennack_cpp.csv");
			res_file << "c,d, X_lower_bound, X_upper_bound" << "\n";
			res_file << cplex.getValue(c[0]) << "," << cplex.getValue(d[0]) << "," << data[0][0] << "," << cplex.getValue((d[1] - d[0]) / (c[0] - c[1])) << "\n";
			for (b = 1; b < Breakpoints - 2; b++) {
				res_file << cplex.getValue(c[b]) << "," << cplex.getValue(d[b]) << "," << cplex.getValue((d[b] - d[b - 1]) / (c[b - 1] - c[b])) << "," << cplex.getValue((d[b + 1] - d[b]) / (c[b] - c[b + 1])) << "\n";
			}
			res_file << cplex.getValue(c[Breakpoints - 2]) << "," << cplex.getValue(d[Breakpoints - 2]) << "," << cplex.getValue((d[Breakpoints - 2] - d[Breakpoints - 3]) / (c[Breakpoints - 3] - c[Breakpoints - 2])) << "," << data[DataPoints - 1][0] << "\n";
			res_file.close();
			// cout << endl;
			// cout << "cmax: " << cmax << ", cmin: " << cmin << endl;
			// cout << "dmax: " << dmax << ", dmin: " << dmin << endl;
			// ofstream file_Ma;
			// ofstream file_M2;
			// file_M2.open("m2.txt");
			// file_Ma.open("Ma.txt");
			// for(int i = 0; i < DataPoints; i++){
			// 	file_M2 << M_2[i] << "\n";
			// 	file_Ma << M_a[i] << "\n";
			// }
			// file_M2.close();
			// file_Ma.close();
		}
	}

	catch (IloException& ex) {
		cerr << "Error: " << ex << endl;
	}
	catch (...) {
		cerr << "Error" << endl;
	}
	env.end();
	return 0;
}

