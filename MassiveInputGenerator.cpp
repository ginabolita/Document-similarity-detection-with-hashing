#include <iostream>
#include <vector>
#include <unordered_set>
#include <string>
#include <queue>
#include <algorithm>
using namespace std;

int main()
{
    int num_docs;
    cin >> num_docs;
    for (int k = 1; k < 15; ++k)
    {
        for (int t = 100; t < 1000; t += 100)
        {
            for (int b = t/10; b < t; b += t/10)
            {
                string out = "python3 prueba3.py --mode virtual --k_values ";
                out += to_string(k);
                out += " --t_values ";
                out += to_string(t);
                out += " --b_values ";
                out += to_string(b);
                out += " --threshold_values 0.5 --num_docs ";
                out += to_string(num_docs);
                out += " --prepare_datasets";
                cout << out << endl;
            }
        }
    }
}
