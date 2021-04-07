#include <vector>
#include <iostream>

/**
 * @brief strukt�ra opisuj�ca rozmiar macierzy 
*/
struct matrix_size {
	/**
	 * @brief wysoko�� macierzy
	*/
	const size_t hight;
	/**
	 * @brief szeroko�� macierzy
	*/
	const size_t width;
};

/**
 * @brief funkcjia inicjalizuje macierz danymi
 * @tparam T typ danych w macierzy 
 * @param size wielko�� macierzy 
 * @param matrix macierz reprezentowana w wektorze 
*/
template <typename T>
void init_matrix(matrix_size size, std::vector<T>& matrix)
{
	for (size_t i = 0; i < size.hight; i++)
	{
		for (size_t k = 0; k < size.width; k++)
		{
			// tymczasowe rozwi�zanie potem chyba lepiej b�dzie to �adowa� z pliku
			matrix[i * size.width+ k] = i * 10 + k;
		}
	}
}

/**
 * @brief funkcjia drukuje macierz na terminalu
 * @tparam T typ danych w macierzy
 * @param size wielko�� macierzy
 * @param matrix macierz reprezentowana w wektorze
*/
template <typename T>
void print_matrix(matrix_size size, std::vector<T>& matrix)
{
	for (size_t i = 0; i < size.hight; i++)
	{
		for (size_t k = 0; k < size.width; k++)
		{
			std::cout << matrix[i * size.width + k] << ", ";
		}
		std::cout << std::endl;
	}
}

