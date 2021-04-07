//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#if defined(FPGA) || defined(FPGA_EMULATOR)
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif
#include <array>
#include <iostream>
#include "utility.cpp"

using namespace cl::sycl;

// Convience data access definitions
constexpr access::mode dp_read = access::mode::read;
constexpr access::mode dp_write = access::mode::write;

// ARRAY type & data size for use in this example
constexpr size_t array_size = 10000;
typedef std::array<int, array_size> IntArray;

//typy dla macierzy, 
constexpr matrix_size m1_size = { 3,3 };
constexpr matrix_size m2_size = { 3,3 };




// output message for runtime exceptions
#define EXCEPTION_MSG \
  "    If you are targeting an FPGA hardware, please ensure that an FPGA board is plugged to the system, \n\
        set up correctly and compile with -DFPGA  \n\
    If you are targeting the FPGA emulator, compile with -DFPGA_EMULATOR.\n"

//************************************
// Function description: initialize the array from 0 to array_size-1
//************************************
void initialize_array(IntArray& a) {
	for (size_t i = 0; i < a.size(); i++) a[i] = i;
}


//************************************
// Function description: create a device queue with the default selector or
// explicit FPGA selector when FPGA macro is defined
//    return: DPC++ queue object
//************************************
queue create_device_queue() {
	// create device selector for the device of your interest
#ifdef FPGA_EMULATOR
  // DPC++ extension: FPGA emulator selector on systems without FPGA card
	INTEL::fpga_emulator_selector dselector;
#elif defined(FPGA)
  // DPC++ extension: FPGA selector on systems with FPGA card
	INTEL::fpga_selector dselector;
#else
  // the default device selector: it will select the most performant device
  // available at runtime.
	default_selector dselector;
#endif

	// create an async exception handler so the program fails more gracefully.
	auto ehandler = [](cl::sycl::exception_list exceptionList) {
		for (std::exception_ptr const& e : exceptionList) {
			try {
				std::rethrow_exception(e);
			}
			catch (cl::sycl::exception const& e) {
				std::cout << "Caught an asynchronous DPC++ exception, terminating the "
					"program."
					<< std::endl;
				std::cout << EXCEPTION_MSG;
				std::terminate();
			}
		}
	};

	try {
		// create the devices queue with the selector above and the exception
		// handler to catch async runtime errors the device queue is used to enqueue
		// the kernels and encapsulates all the states needed for execution
		queue q(dselector, ehandler);

		return q;
	}
	catch (cl::sycl::exception const& e) {
		// catch the exception from devices that are not supported.
		std::cout << "An exception is caught when creating a device queue."
			<< std::endl;
		std::cout << EXCEPTION_MSG;
		std::terminate();
	}
}



/**
 * @brief funkcjia mno¿y przez siebie dwie macierze
 * @tparam T typ danych przechowywanych przez macierz
 * @param m1_size wielkoœæ macierzy 1
 * @param m2_size wielkoœæ macierzy 2
 * @param m1 macierz 1
 * @param m2 macierz 2
 * @param result macierz do której zostanie wpisany wynik
*/
template <typename T >
void matrix_mult(const matrix_size& m1_size, const matrix_size& m2_size, const std::vector<T>& m1, const std::vector<T>& m2, std::vector<T>& result)
{
	if (m1_size.hight != m2_size.width )
	{
		throw "matrix have wrong sizes";
	}
	else if (m1.size() != m1_size.width * m1_size.hight)
	{
		throw "matrix m1 is wrongly defined";
	}
	else if (m2.size() != m2_size.width * m2_size.hight)
	{
		throw "matrix m2 is wrongly defined";
	}
	else if (m1_size.hight * m2_size.width != result.size())
	{
		throw "resutl matrix is wrongly defined";
	}
	// @TODO zmieniæ te wyj¹teki na coœ lepszego
	queue q = create_device_queue();
#ifdef DEBUG
	std::cout << "Device: " << q.get_device().get_info<info::device::name>()
		<< std::endl;
#endif // DEBUG
	matrix_size result_size = { m1_size.hight,m2_size.width };

	range<1> num_items{ m1_size.width * m1_size.hight };
	range<1> num_result{ m1_size.hight * m2_size.width };

	buffer<T, 1> m1_buff(m1.data(), num_items);// ? const
	buffer<T, 1> m2_buff(m2.data(), num_items);// ? const
	buffer<T, 1> result_buff(result.data(), num_result);

	q.submit([&](handler& h) {
		auto m1_accesor = m1_buff.template get_access<dp_read>(h);
		auto m2_accesor = m2_buff.template get_access<dp_read>(h);

		auto result_accesor = result_buff.template get_access<access::mode::read_write>(h); // czu tutaj wystarczy tylko obs³uga write ?

		h.parallel_for(num_result, [=](id<1> i) {
			auto temp = i[0];
			size_t w = i % result_size.width;
			auto h = i != id<1>() ? (i[0] - w) / result_size.width : 0;
			for (size_t k = 0; k < m1_size.width; k++) // czy to te¿ powinienem zrównolegliæ ?
			{
				result_accesor[i] += m1_accesor[get_coord(m1_size, h, k)] * m2_accesor[get_coord(m2_size, k, w)];
			}
			});
		});
}


int main() {


	std::vector<int> m1(m1_size.hight * m1_size.width, 0);
	std::vector<int> m2(m2_size.hight * m2_size.width, 0);
	std::vector<int> result(m1_size.hight * m2_size.width, 0);
	std::cout << "sdafsdsdfsdfsdfsdffdsdfsdfdfm1:" << std::endl;

	init_matrix(m1_size, m1);
	init_matrix(m2_size, m2);
	std::cout << "m1:" << std::endl;
	print_matrix(m1_size, m1);

	std::cout << "m2:" << std::endl;
	print_matrix(m2_size, m2);

	matrix_mult(m1_size, m2_size, m1, m2, result);

	std::cout << "wynik:" << std::endl;
	print_matrix({ m1_size.hight,m2_size.width }, result);

	return 0;
}