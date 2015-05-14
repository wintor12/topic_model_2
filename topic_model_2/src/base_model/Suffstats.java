package base_model;


import java.util.Random;

import org.apache.commons.math3.random.MersenneTwister;

public class Suffstats {
	double[][] class_word;
	double[] class_total;
	double alpha_suffstas;
	int num_topics;
	int num_terms;
	int num_docs;
	Corpus corpus;
	public double beta;
	MersenneTwister mt = new MersenneTwister(32);
	
	public Suffstats(Model model, Corpus corpus, double beta)
	{
		num_topics = model.num_topics;
		num_terms = model.num_terms;
		class_total = new double[num_topics];
		class_word = new double[num_topics][num_terms];
		this.corpus = corpus;
		this.beta = beta;
	}
	
	public void gibbs_initialize_ss()
	{
		int M = corpus.docs.length;
		int K = num_topics;
		int V = corpus.voc.size();
		int[][] nmk = new int[M][K];
		int[][] nkt = new int[K][V];    //V is ids of vocabulary
		int[] nmkSum = new int[M];
		int[] nktSum = new int[K];		
		
		for(int m = 0; m < M; m++)
		{
			Document doc = corpus.docs[m];

			for(int n = 0; n < doc.length; n++)
			{
				int initTopic = (int)(mt.nextDouble() * K);// From 0 to K - 1
				doc.z[n] = initTopic;
				//number of words in doc m assigned to topic initTopic add 1
				nmk[m][initTopic]++;
				//number of terms doc[m][n] assigned to topic initTopic add 1
				nkt[initTopic][doc.ids[n]]++;
				// total number of words assigned to topic initTopic add 1
				nktSum[initTopic]++;
			}
			// total number of words in document m is N
		    nmkSum[m] = doc.length;
		}
		for (int k = 0; k < num_topics; k++)
	    {
			class_total[k] = nktSum[k] + V*beta;
	        for (int n = 0; n < num_terms; n++)
	        {
	            class_word[k][n] = nkt[k][n] + beta;
	        }
	    }
	}
	
	public void beta_initialize_ss()
	{
		int k, n;
		for (k = 0; k < num_topics; k++)
	    {
			class_total[k] = beta*corpus.voc.size();
	        for (n = 0; n < num_terms; n++)
	        {
	            class_word[k][n] = beta;	            
	        }
	    }
		this.num_docs = 0;
		this.alpha_suffstas = 0;
	}
	
	//Random initialize word-topic joint probability
	public void random_initialize_ss()
	{
//		MersenneTwister mt = new MersenneTwister(10);   //set seed
		Random rand = new Random();   
		int k, n;
		for (k = 0; k < num_topics; k++)
	    {
	        for (n = 0; n < num_terms; n++)
	        {
	            class_word[k][n] += 1.0/num_terms + rand.nextDouble();
	            class_total[k] += class_word[k][n];
	        }
	    }
	}
	
	public void zero_initialize_ss()
	{
		int k, n;
		for (k = 0; k < num_topics; k++)
	    {
			class_total[k] = 0;
	        for (n = 0; n < num_terms; n++)
	        {
	            class_word[k][n] = 0;	            
	        }
	    }
		this.num_docs = 0;
		this.alpha_suffstas = 0;
	}
}

