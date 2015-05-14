package base_model;


import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;

import org.apache.commons.io.FileUtils;
import org.apache.commons.math3.special.Gamma;

public class Model {
	
	public double[][] log_prob_w; //log beta   K*V
	public double alpha;
	public int num_topics;
	int num_terms;
	public static double NEWTON_THRESH = 1e-4;
	public static int MAX_ALPHA_ITER = 50;
	
	public Model(int num_topics, int num_terms, double alpha)
	{
		this.num_terms = num_terms;
		this.num_topics = num_topics;
		log_prob_w = new double[num_topics][num_terms]; 
		this.alpha = alpha;
	}
	
	//if init is true, then we use the initial alpha,
	//else update alpha
	public void mle(Suffstats ss, boolean init)
	{
		int k, w;
		//Update beta from sufficient statistics
		for (k = 0; k < num_topics; k++)
	    {
	        for (w = 0; w < num_terms; w++)
	        {	        	
	        	//log p(w|k) = log(p(w,k)/p(k)) = log p(w,k) - log p(w)
//	            if (ss.class_word[k][w] > 0)
//	                this.log_prob_w[k][w] = Math.log(ss.class_word[k][w]) - Math.log(ss.class_total[k]);
//	            else
//	                this.log_prob_w[k][w] = -100;
	        	if(ss.class_word[k][w] == 0)
	        		System.out.println("===============error");
	        	this.log_prob_w[k][w] = Math.log(ss.class_word[k][w]) - Math.log(ss.class_total[k]);
	        }
	    }
		
		//Update alpha
		if(init == false)
			this.alpha = opt_alpha(ss.alpha_suffstas, ss.num_docs, this.num_topics);
	}
	
	/**
	 * newtons method
	 * @D ss.num_topics
	 * @K topics
	 */
	double opt_alpha(double ss, int D, int K)
	{
	    double a, log_a, init_a = 100;
	    double f, df, d2f;
	    int iter = 0;

	    log_a = Math.log(init_a);
	    do
	    {
	        iter++;
	        a = Math.exp(log_a);
	        if (Double.isNaN(a))
	        {
	            init_a = init_a * 10;
	            System.out.println("warning : alpha is nan; new init = " +  init_a);
	            a = init_a;
	            log_a = Math.log(a);
	        }
	        f = D * (Gamma.logGamma(K * a) - K * Gamma.logGamma(a)) + (a - 1) * ss;
	        df = D * (K * Gamma.digamma(K * a) - K * Gamma.digamma(a)) + ss;
	        d2f = D * (K * K * Gamma.trigamma(K * a) - K * Gamma.trigamma(a));
	        log_a = log_a - df/(d2f * a + df);
//	        System.out.format("alpha maximization : %5.5f   %5.5f\n", f, df);
	    }
	    while ((Math.abs(df) > NEWTON_THRESH) && (iter < MAX_ALPHA_ITER));
	    return(Math.exp(log_a));
	}
	
	/**
	 * save current beta and other parameters
	 * @param path         The folder where beta saved
	 * @param filename     The name of beta(usually like "init" or the number of iteration)
	 */
	public void save_lda_model(String filename)
	{
		DecimalFormat df = new DecimalFormat("#.##");
		StringBuilder sb = new StringBuilder();
		int k, w;
		for (k = 0; k < num_topics; k++)
	    {
	        for (w = 0; w < num_terms; w++)
	        {
//	        	sb.append(df.format(Math.exp(this.log_prob_w[k][w])) + " ");
	        	sb.append(Math.exp(this.log_prob_w[k][w]) + " ");
	        }
	        sb.append("\n");
        }
		try {
			FileUtils.writeStringToFile(new File(filename + "_beta"), sb.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		StringBuilder sb2 = new StringBuilder();
		sb2.append("num_topics: " + this.num_topics);
		sb2.append("\n");
		sb2.append("num_terms: " + this.num_terms);
		sb2.append("\n");
		sb2.append("alpha: " + df.format(this.alpha));
		sb2.append("\n");
		try {
			FileUtils.writeStringToFile(new File(filename + "_other"), sb2.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
}

