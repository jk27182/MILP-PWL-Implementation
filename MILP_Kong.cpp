// -------------------------------------------------------------- -*- C++ -*-

#include <ilcplex/ilocplex.h>
ILOSTLBEGIN
using namespace std;

typedef IloArray<IloNumVarArray> NumVarMatrix;
typedef IloArray<IloNumArray>    NumMatrix;
typedef IloArray<IloBoolVarArray> BoolVarMatrix;
typedef IloArray<IloNumExprArray> ArrayMatrix;

std::string filename = "data/MpStorage50.txt";
const IloInt Breakpoints = 4;

// 1 = LInf, 2=, 3=.
IloInt q = 2; //DistanceMetric

int
main()
{
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
	cout << "Number of Data Points = " << DataPoints << endl;
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
	for (i = 0; i < DataPoints; i++) {
		for (int j = 0; j < i; j++) {
			if ((data[i][1] - data[j][1]) / (data[i][0] - data[j][0]) > cmax) {
				cmax = (data[i][1] - data[j][1]) / (data[i][0] - data[j][0]);
			}
			if ((data[i][1] - data[j][1]) / (data[i][0] - data[j][0]) < cmin) {
				cmin = (data[i][1] - data[j][1]) / (data[i][0] - data[j][0]);
			}
		}
	}
	cout << cmin << " " << cmax << endl;

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
	}

	IloEnv env;
	try {
		//------------------------------------------------------Variable Definitions------------------------------------------
		IloInt i, b;
		NumVarMatrix Y(env, DataPoints);
		for (i = 0; i < DataPoints; i++) {
			Y[i] = IloNumVarArray(env, Breakpoints - 1, -IloInfinity, IloInfinity);
		}
		IloNumVarArray A(env, Breakpoints - 1, cmin, cmax);
		IloNumVarArray B(env, Breakpoints - 1, dmin, dmax);

		IloNumVarArray E(env, DataPoints, 0, IloInfinity);
		IloNumVar E1(env, 0, IloInfinity);
		NumVarMatrix Pplus(env, DataPoints + 1);
		NumVarMatrix Pminus(env, DataPoints + 1);
		NumVarMatrix Qplus(env, DataPoints + 1);
		NumVarMatrix Qminus(env, DataPoints + 1);

		BoolVarMatrix U(env, DataPoints + 1);
		BoolVarMatrix V(env, DataPoints + 1);

		BoolVarMatrix Z(env, DataPoints + 1);
		BoolVarMatrix ZF(env, DataPoints + 1);
		BoolVarMatrix ZL(env, DataPoints + 1);

		for (i = 0; i < DataPoints + 1; i++) {
			Pplus[i] = IloNumVarArray(env, Breakpoints - 1, 0, IloInfinity);
			Pminus[i] = IloNumVarArray(env, Breakpoints - 1, 0, IloInfinity);
			Qplus[i] = IloNumVarArray(env, Breakpoints - 1, 0, IloInfinity);
			Qminus[i] = IloNumVarArray(env, Breakpoints - 1, 0, IloInfinity);

			U[i] = IloBoolVarArray(env, Breakpoints - 2);
			V[i] = IloBoolVarArray(env, Breakpoints - 2);

			Z[i] = IloBoolVarArray(env, Breakpoints - 1);
			ZF[i] = IloBoolVarArray(env, Breakpoints - 1);
			ZL[i] = IloBoolVarArray(env, Breakpoints - 1);
		}

		//-------------------------------------------------------Adding Constraints-------------------------------------------------------------
		IloModel model(env);

		//(3)
		for (int i = 0; i < DataPoints; i++) {
			IloNumExpr sum_over_bZ(env);
			for (int b = 0; b < Breakpoints - 1; b++) {
				sum_over_bZ += Z[i][b];
			}
			model.add(sum_over_bZ == 1);
		}


		for (i = 0; i < DataPoints; i++) {
			for (b = 0; b < Breakpoints - 1; b++) {
				//(5)
				//model.add(Y[i][b] == data[i][0] * A[b] + B[b]);

				//(6)-(7)
				if (q == 1) {
					model.add(E1 >= (data[i][0] * A[b] + B[b]) - data[i][1] - M_a[i] * (1 - Z[i][b]));
					model.add(E1 >= data[i][1] - (data[i][0] * A[b] + B[b]) - M_a[i] * (1 - Z[i][b]));
				}
				else {
					model.add(E[i] >= (data[i][0] * A[b] + B[b]) - data[i][1] - M_a[i] * (1 - Z[i][b]));
					model.add(E[i] >= data[i][1] - (data[i][0] * A[b] + B[b]) - M_a[i] * (1 - Z[i][b]));
				}
			}
		}

		//(17)
		for (b = 0; b < Breakpoints - 1; b++) {
			model.add(Z[0][b] == ZF[0][b]);
		}
		for (i = 1; i < DataPoints; i++) {
			for (b = 0; b < Breakpoints - 1; b++) {
				model.add(Z[i][b] == Z[i - 1][b] + ZF[i][b] - ZL[i - 1][b]);
			}
		}

		//(18)-(19)
		for (b = 0; b < Breakpoints - 1; b++) {
			IloNumExpr sum_over_iZL(env);
			IloNumExpr sum_over_iZF(env);
			for (i = 0; i < DataPoints; i++) {
				sum_over_iZL += ZL[i][b];
				sum_over_iZF += ZF[i][b];
			}
			model.add(sum_over_iZL == 1);
			model.add(sum_over_iZF == 1);
		}

		//(20)-(21)
		for (i = 0; i < DataPoints - 1; i++) {
			for (b = 0; b < Breakpoints - 2; b++) {
				IloNumExpr partsum_ZFa(env);
				IloNumExpr partsum_ZFb(env);
				IloNumExpr partsum_ZLa(env);
				IloNumExpr partsum_ZLb(env);
				for (int j = 0; j < i + 1; j++) {
					partsum_ZFa += ZF[j][b];
					partsum_ZFb += ZF[j][b + 1];
					partsum_ZLa += ZL[j][b];
					partsum_ZLb += ZL[j][b + 1];
				}
				model.add(partsum_ZFa >= partsum_ZFb);
				model.add(partsum_ZLa >= partsum_ZLb);
			}
		}

		//(22)-(23)
		for (i = 0; i < DataPoints; i++) {
			for (b = 0; b < Breakpoints - 1; b++) {
				model.add(ZF[i][b] <= Z[i][b]);
				model.add(ZL[i][b] <= Z[i][b]);
			}
		}

		//(25)-(31)
		for (i = 0; i < DataPoints - 1; i++) {
			for (b = 0; b < Breakpoints - 2; b++) {

				//(25)-(26)
				model.add(data[i][0] * A[b + 1] + B[b + 1] - (data[i][0] * A[b] + B[b]) == Pplus[i][b] - Pminus[i][b]);
				model.add(data[i + 1][0] * A[b] + B[b] - (data[i + 1][0] * A[b + 1] + B[b + 1]) == Qplus[i + 1][b + 1] - Qminus[i + 1][b + 1]);

				//(27)-(30)
				model.add(Pplus[i][b] <= M_a[i] * (1 - U[i][b]));
				model.add(Qplus[i + 1][b + 1] <= M_a[i] * (1 - U[i][b]));
				model.add(Pminus[i][b] <= M_a[i] * (1 - V[i][b]));
				model.add(Qminus[i + 1][b + 1] <= M_a[i] * (1 - V[i][b]));

				//(31)
				model.add(U[i][b] + V[i][b] == ZL[i][b]);
			}
		}

		//-------------------------------------------------Model Objective--------------------------------------

		IloNumExpr objective(env);
		if (q == 1) {
			model.add(IloMinimize(env, E1));
		}
		else {
			for (int i = 0; i < DataPoints; i++) {
				if (q == 2) {
					objective += E[i];
				}
				else if (q == 3) {
					objective += E[i] * E[i];
				}
			}
			model.add(IloMinimize(env, objective));
		}

		//-------------------------------------------------Solve the Model -------------------------------------------------

		IloCplex cplex(model);
		cplex.solve();

		cout << "Solution status: " << cplex.getStatus() << endl;
		cout << "Objective Value = " << cplex.getObjValue() << endl;

		IloBool WriteToFile = IloFalse;
		if (WriteToFile) {
			ofstream solfile;
			solfile.open("Functions.txt");
			solfile << "Value = " << cplex.getObjValue() << endl << "Affine Functions: " << endl;
			solfile << cplex.getValue(A[0]) << " * X + " << cplex.getValue(B[0]) << " (" << data[0][0] << " <= X <= " << cplex.getValue((B[1] - B[0]) / (A[0] - A[1])) << ")" << endl;
			for (b = 1; b < Breakpoints - 2; b++) {
				solfile << cplex.getValue(A[b]) << " * X + " << cplex.getValue(A[b]) << " (" << cplex.getValue((B[b] - B[b - 1]) / (A[b - 1] - A[b])) << " <= X <= " << cplex.getValue((B[b + 1] - B[b]) / (A[b] - A[b + 1])) << ")" << endl;
			}
			solfile << cplex.getValue(A[Breakpoints - 2]) << " * X + " << cplex.getValue(B[Breakpoints - 2]) << " (" << cplex.getValue((B[Breakpoints - 2] - B[Breakpoints - 3]) / (A[Breakpoints - 3] - A[Breakpoints - 2])) << " <= X <= " << data[DataPoints - 1][0] << ")" << endl;
			solfile << endl;
			solfile.close();
		}

		IloBool Print = IloTrue;
		if (Print) {
			ofstream res_file;
			res_file.open("res_kong.csv");
			res_file << "c,d, X_lower_bound, X_upper_bound" << "\n";
			res_file << cplex.getValue(A[0]) << "," << cplex.getValue(B[0]) << "," << data[0][0] << "," << cplex.getValue((B[1] - B[0]) / (A[0] - A[1])) << "\n";
			for (b = 1; b < Breakpoints - 2; b++) {
				res_file << cplex.getValue(A[b]) << "," << cplex.getValue(B[b]) << "," << cplex.getValue((B[b] - B[b - 1]) / (A[b - 1] - A[b])) << "," << cplex.getValue((B[b + 1] - B[b]) / (A[b] - A[b + 1])) << "\n";
			}
			res_file << cplex.getValue(A[Breakpoints - 2]) << "," << cplex.getValue(B[Breakpoints - 2]) << "," << cplex.getValue((B[Breakpoints - 2] - B[Breakpoints - 3]) / (A[Breakpoints - 3] - A[Breakpoints - 2])) << "," << data[DataPoints - 1][0] << "\n";
			res_file.close();

			cout << "Value = " << cplex.getObjValue() << endl << "Affine Functions: " << endl;
			cout << cplex.getValue(A[0]) << " * X + " << cplex.getValue(B[0]) << " (" << data[0][0] << " <= X <= " << cplex.getValue((B[1] - B[0]) / (A[0] - A[1])) << ")" << endl;
			for (b = 1; b < Breakpoints - 2; b++) {
				cout << cplex.getValue(A[b]) << " * X + " << cplex.getValue(A[b]) << " (" << cplex.getValue((B[b] - B[b - 1]) / (A[b - 1] - A[b])) << " <= X <= " << cplex.getValue((B[b + 1] - B[b]) / (A[b] - A[b + 1])) << ")" << endl;
			}
			cout << cplex.getValue(A[Breakpoints - 2]) << " * X + " << cplex.getValue(B[Breakpoints - 2]) << " (" << cplex.getValue((B[Breakpoints - 2] - B[Breakpoints - 3]) / (A[Breakpoints - 3] - A[Breakpoints - 2])) << " <= X <= " << data[DataPoints - 1][0] << ")" << endl;
			cout << endl;
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
