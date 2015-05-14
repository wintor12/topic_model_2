package base_model;

import java.util.Comparator;


public class Tools {
	
	/*
	 * given log(a) and log(b), return log(a + b)
	 *
	 */
	public static double log_sum(double log_a, double log_b) {
		double v;

		if (log_a < log_b) {
			v = log_b + Math.log(1 + Math.exp(log_a - log_b));
		} else {
			v = log_a + Math.log(1 + Math.exp(log_b - log_a));
		}
		return (v);
	}
	
	public static double evalChebyStar (double a[], int n, double x)  {
	      if (x > 1.0 || x < 0.0)
	         System.err.println ("Shifted Chebychev polynomial evaluated " +
	                             "at x outside [0, 1]");
	      final double xx = 2.0*(2.0*x - 1.0);
	      double b0 = 0.0, b1 = 0.0, b2 = 0.0;
	      for (int j = n; j >= 0; j--) {
	         b2 = b1;
	         b1 = b0;
	         b0 = xx*b1 - b2 + a[j];
	      }
	      return (b0 - b2)/2.0;
   }
	
	public static double tetragamma (double x) {
	      double y, sum;

	      if (x < 0.5) {
	         y = (1.0 - x) - Math.floor (1.0 - x);
	         sum = Math.PI / Math.sin (Math.PI * y);
	         return 2.0 * Math.cos (Math.PI * y) * sum * sum * sum +
	               tetragamma (1.0 - x);
	      }

	      if (x >= 20.0) {
	         // Asymptotic series
	         y = 1.0/(x*x);
	         sum = y*(0.5 - y*(1.0/6.0 - y*(1.0/6.0 - y*(0.3 - 5.0/6.0*y))));
	         sum += 1.0 + 1.0/x;
	         return -sum*y;
	      }

	      int i;
	      int p = (int) x;
	      y = x - p;
	      sum = 0.0;

	      if (p > 3) {
	         for (i = 3; i < p; i++)
	            sum += 1.0 / ((y + i) * (y + i) * (y + i));

	      } else if (p < 3) {
	         for (i = 2; i >= p; i--)
	            sum -= 1.0 / ((y + i) * (y + i) * (y + i));
	      }

	      /* Chebyshev coefficients for tetragamma (x + 3), 0 <= x <= 1. In Yudell
	         Luke: The special functions and their approximations, Vol. II,
	         Academic Press, p. 301, 1969. */
	      final int N = 16;
	      final double A[] = { -0.11259293534547383037*2.0, 0.03655700174282094137,
	         -0.00443594249602728223, 0.00047547585472892648,
	         -4.747183638263232e-5, 4.52181523735268e-6, -4.1630007962011e-7,
	         3.733899816535e-8, -3.27991447410e-9, 2.8321137682e-10,
	         -2.410402848e-11, 2.02629690e-12, -1.6852418e-13, 1.388481e-14,
	         -1.13451e-15, 9.201e-17, -7.41e-18, 5.9e-19, -5.0e-20 };

	      return 2.0 * sum + evalChebyStar (A, N, y);
	   }

	
	class ArrayIndexComparator implements Comparator<Integer>
	{
		double[] array;
		public ArrayIndexComparator(double[] array)
	    {
	        this.array = array;
	    }
		
		public Integer[] createIndexArray() 
		{
			Integer[] indexes = new Integer[array.length];
			for (int i = 0; i < array.length; i++) {
				indexes[i] = i; 
			}
			return indexes;
		}


		@Override
		public int compare(Integer index1, Integer index2) {
			if(array[index1] < array[index2])
				return 1;
			else if(array[index1] > array[index2])
				return -1;
			else
				return 0;
		}

	}
	
}

