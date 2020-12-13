//============================================================================
// Name        : parallel.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <string>
#include <chrono>
#include <math.h>
#include <omp.h>

#define MAX 2

using namespace std;


typedef double (* vEvalCall)(double A, double B, int n);

struct named_eval_call {
	string name;
	vEvalCall call;
};


double f(double x)
{
	double sin_value = sin(1 / x);
	return 1 / (x * x) * sin_value * sin_value;
}


double serial_eval(double A, double B, int n)
{
	double step_size = (B - A) / n;
	double result = 0;

	for (int i = 1; i < n - 1; i++)
	{
		double x_i = A + step_size * i;
		result += f(x_i);
	}

	return step_size * (result + (f(A) + f(B)) / 2);
}


double atomic_eval(double A, double B, int n)
{
	double step_size = (B - A) / n;
	double result = 0;

	#pragma omp parallel for num_threads(MAX)
	for (int i = 1; i < n - 1; i++)
	{
		double x_i = A + step_size * i;
		double f_value = f(x_i);
		#pragma omp atomic
		result += f_value;
	}

	return step_size * (result + (f(A) + f(B)) / 2);
}


double critical_eval(double A, double B, int n)
{
	double step_size = (B - A) / n;
	double result = 0;

	#pragma omp parallel for num_threads(MAX)
	for (int i = 1; i < n - 1; i++)
	{
		double x_i = A + step_size * i;
		double f_value = f(x_i);
		#pragma omp critical
		{
			result += f_value;
		}
	}

	return step_size * (result + (f(A) + f(B)) / 2);
}


double locks_eval(double A, double B, int n)
{
	double step_size = (B - A) / n;
	double result = 0;
	omp_lock_t lock;
	omp_init_lock(&lock);

	#pragma omp parallel for num_threads(MAX)
	for (int i = 1; i < n - 1; i++)
	{
		double x_i = A + step_size * i;
		double f_value = f(x_i);
		omp_set_lock(&lock);
		result += f_value;
		omp_unset_lock(&lock);
	}

	omp_destroy_lock(&lock);
	return step_size * (result + (f(A) + f(B)) / 2);
}


double reduction_eval(double A, double B, int n)
{
	double step_size = (B - A) / n;
	double result = 0;

	#pragma omp parallel for reduction(+:result) num_threads(MAX)
	for (int i = 1; i < n - 1; i++)
	{
		double x_i = A + step_size * i;
		double f_value = f(x_i);
		result += f_value;
	}

	return step_size * (result + (f(A) + f(B)) / 2);
}


void box(string name, double result, unsigned int time) {
	cout << name << endl;
	cout << "Result: " << result << endl;
	cout << "Runtime: " << time << endl;
	cout << endl;
}


void run_benchmark(vEvalCall eval, string name, double A, double B, int n) {
	auto start = chrono::high_resolution_clock::now();
	double result = eval(A, B, n);
	auto stop = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
	box(name, result, duration.count());
}


int main()
{
	named_eval_call calls[5] = {
			{ "Serial", &serial_eval },
			{ "Atomic", &atomic_eval },
			{ "Critical", &critical_eval },
			{ "Lock", &locks_eval },
			{ "Reduction", &reduction_eval }
	};

	for (int i = 0; i < 5; i++) {
		run_benchmark(calls[i].call, calls[i].name, 0.00001, 0.001, 10000000);
	}
}
