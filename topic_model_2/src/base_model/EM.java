package base_model;

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import org.apache.commons.io.FileUtils;
import org.apache.commons.math3.special.Gamma;

import base_model.Tools.ArrayIndexComparator;

public class EM {

	public int num_topics; // topic numbers   K
	public int VAR_MAX_ITER;
	public double VAR_CONVERGED;
	public int EM_MAX_ITER;
	public double EM_CONVERGED;
	public double alpha;     //dirichlet prior of topic mixture 
	public double beta;      //dirichlet prior of word mixture, used for gibbs initialize suf statistic
	public String path = ""; //running path
	public String path_res = ""; //result path
	public Corpus corpus;

	public EM(String path, String path_res, int num_topics, Corpus corpus, double beta) {
		this.path = path;
		this.path_res = path_res;
		this.num_topics = num_topics; // topic numbers
		this.VAR_MAX_ITER = 20;
		this.VAR_CONVERGED = 1e-6;
		this.EM_MAX_ITER = 100;
		this.EM_CONVERGED = 1e-4;
		this.alpha = 0.1;
		this.corpus = corpus;
		this.beta = beta;
	}

	public EM(String path, String path_res, int num_topics, Corpus corpus, int vAR_MAX_ITER, double vAR_CONVERGED,
			int eM_MAX_ITER, double eM_CONVERGED, double alpha, double beta) {
		this.path_res = path_res;
		this.num_topics = num_topics;
		this.VAR_MAX_ITER = vAR_MAX_ITER;
		this.VAR_CONVERGED = vAR_CONVERGED;
		this.EM_MAX_ITER = eM_MAX_ITER;
		this.EM_CONVERGED = eM_CONVERGED;
		this.alpha = alpha;
		this.path = path;
		this.corpus = corpus;
		this.beta = beta;
	}
	
	public void run_em(String type)
	{ 
		String path_model = new File(path_res, "model").getAbsolutePath();
		
		Model model = new Model(num_topics, corpus.num_terms, alpha);
		Suffstats ss = new Suffstats(model, corpus, beta);
		//Random initialize joint probability of p(w, k), and compute p(k) by sum over p(w, k)
		ss.random_initialize_ss();
//		ss.gibbs_initialize_ss();
		model.mle(ss, true); //get initial beta
		
//		model.save_lda_model(new File(path_model, "init").getAbsolutePath());		
		
		//run EM 		
		double likelihood, likelihood_old = 0, converged = 1;
		int i = 0;
		StringBuilder sb = new StringBuilder(); //output likelihood and converged
		while(((converged < 0) || (converged > EM_CONVERGED) || (i <= 2)) && (i <= EM_MAX_ITER))
		{
			i++;
			System.out.println("**** em iteration " + i + "****");
			likelihood = 0;
			ss.beta_initialize_ss();
//			ss.zero_initialize_ss();
			//E step
			for(int d = 0; d < corpus.num_docs; d++)
			{
				if(d%1000 == 0)
					System.out.println("document " + d);
				
				//Initialize gamma and phi to zero for each document
				corpus.docs[d].gamma = new double[model.num_topics];
				corpus.docs[d].phi = new double[corpus.maxLength()][num_topics];
				
				//Compute gamma, phi of each document, and update ss
				//Sum up likelihood of each document
				likelihood += doc_e_step(corpus.docs[d], model, ss); 
			}

			// M step
			//Update Model.beta and Model.alpha using ss
			model.mle(ss, false);
			
			// check for convergence
	        converged = (likelihood_old - likelihood) / likelihood_old;
	        if (converged < 0) 
	        	VAR_MAX_ITER = VAR_MAX_ITER * 2;
	        likelihood_old = likelihood;
	        	        
	        
	        // output model, likelihood and gamma
	        sb.append(likelihood +"\t" + converged + "\n");
//	        model.save_lda_model(new File(path_model, i + "").getAbsolutePath());
//	        save_gamma(model, new File(path_model, i + "_gamma"));
//	        if(type.equals("GTRF"))
//	        	save_doc_para(new File(path_model, i + "_doc"));
//	        if(type.equals("MGTRF"))
//				save_doc_para2(new File(path_model, i + "_doc"));
		}		
		File likelihood_file = new File(path_res, "likelihood");
		try {
			FileUtils.writeStringToFile(likelihood_file, sb.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		//output the final model
		model.save_lda_model(new File(path_res, "final").getAbsolutePath());
		save_gamma(model, new File(path_res, "final_gamma"));
		if(type.equals("GTRF"))
			save_doc_para(new File(path_res, "final_doc"));
		if(type.equals("MGTRF"))
			save_doc_para2(new File(path_res, "final_doc"));
		
		for(int d = 0; d < corpus.num_docs; d++)
		{
			if(d%1000 == 0)
				System.out.println("final e step document " + d);
			lda_inference(corpus.docs[d], model);
			Document doc = corpus.docs[d];
			doc.theta = new double[num_topics];
			double gamma_sum = 0;
			for(int k = 0; k < num_topics; k++)
			{
				doc.theta[k] = Gamma.digamma(doc.gamma[k]);
				gamma_sum += doc.gamma[k];
			}
			double diggamma_sumgamma =  Gamma.digamma(gamma_sum);
			double max = Integer.MIN_VALUE;
			for(int k = 0; k < num_topics; k++)
			{
				doc.theta[k] -= diggamma_sumgamma;
				if(doc.theta[k] > max)
				{
					max = doc.theta[k];
					doc.cluster = k;
				}
			}
		}
		
		//Save top words of each topic among corpus
		int[][] topwords = save_top_words_corpus(20, model, new File(path_res, "top_words_corpus"));
		//Evaluation
//		computePerplexity_e_theta(model, topwords);
//		computePerplexity_e_theta(model);
//		computePerplexity_gibbs(model, topwords);
//		computePerplexity_gibbs(model);
		
//		pred_dist(model, 0.8);
		
		double nmi = computNMI();
		System.out.println(nmi);
		try {
			File all_eval = new File(path, "eval");
			String s = path_res + " : " + nmi + "\n";
			FileUtils.writeStringToFile(all_eval, s, true);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public double doc_e_step(Document doc, Model model, Suffstats ss)
	{
		double likelihood = 0.0;
		likelihood = lda_inference(doc, model);
		
		// update sufficient statistics
		double gamma_sum = 0;
		for(int k = 0; k < model.num_topics; k++)
		{
			gamma_sum += doc.gamma[k];
			ss.alpha_suffstas += Gamma.digamma(doc.gamma[k]);
		}
		ss.alpha_suffstas -= model.num_topics * Gamma.digamma(gamma_sum);
		for (int n = 0; n < doc.length; n++)
	    {
	        for (int k = 0; k < model.num_topics; k++)
	        {
	            ss.class_word[k][doc.ids[n]] += doc.counts[n]*doc.phi[n][k];
	            ss.class_total[k] += doc.counts[n]*doc.phi[n][k];
	        }
	    }

	    ss.num_docs += 1;
		return likelihood;
	}
	
	public double lda_inference(Document doc, Model model)
	{		
	    double likelihood = 0, likelihood_old = 0;
	    double[] digamma_gam = new double[model.num_topics];
	    
	    // compute posterior dirichlet
	    
	    //initialize varitional parameters gamma and phi
	    for (int k = 0; k < model.num_topics; k++)
	    {
	        doc.gamma[k] = model.alpha + (doc.total/((double) model.num_topics));
	        //compute digamma gamma for later use
	        digamma_gam[k] = Gamma.digamma(doc.gamma[k]);
	        for (int n = 0; n < doc.length; n++)
	            doc.phi[n][k] = 1.0/model.num_topics;
	    }
	    
	    double converged = 1;
	    int var_iter = 0;
	    double[] oldphi = new double[model.num_topics];  //????
	    while (converged > VAR_CONVERGED && var_iter < VAR_MAX_ITER)
	    {
	    	var_iter++;
	    	for(int k = 0; k < num_topics; k++)
	    	{
	        	doc.gamma[k] = 0;
	    	}
//	    	System.out.println("var_iter: " + var_iter);
	    	for(int n = 0; n < doc.length; n++)
	    	{
	    		double phisum = 0;
	    		for(int k = 0; k < model.num_topics; k++)
	    		{
	    			oldphi[k] = doc.phi[n][k];
	    			//phi = beta * exp(digamma(gamma)) -> log phi = log (beta) + digamma(gamma)
	    			doc.phi[n][k] = model.log_prob_w[k][doc.ids[n]] + digamma_gam[k];
	    			if (k > 0)
	                    phisum = Tools.log_sum(phisum, doc.phi[n][k]);
	                else
	                    phisum = doc.phi[n][k]; // note, phi is in log space
	    		}	    		
	    		for (int k = 0; k < model.num_topics; k++)
	            {
	    			//Normalize phi, exp(log phi - log phisum) = phi/phisum
	                doc.phi[n][k] = Math.exp(doc.phi[n][k] - phisum);
//	                doc.gamma[k] += doc.counts[n]*(doc.phi[n][k] - oldphi[k]);
	                doc.gamma[k] += doc.counts[n]*doc.phi[n][k];
//	                digamma_gam[k] = doc.gamma[k] > 0? Gamma.digamma(doc.gamma[k]):Gamma.digamma(0.1);
	            }

	    	}
	    	for(int k = 0; k < num_topics; k++)
	    	{
	    		doc.gamma[k] += model.alpha;
	    		digamma_gam[k] = Gamma.digamma(doc.gamma[k]);
	    	}
	    	likelihood = compute_likelihood(doc, model);
//		    System.out.println("likelihood: " + likelihood);		    
		    converged = (likelihood_old - likelihood) / likelihood_old;
//		    System.out.println(converged);
	        likelihood_old = likelihood;
	    }
	    
	    return likelihood;
	}
	
	public double compute_likelihood(Document doc, Model model)
	{
		double likelihood = 0, gamma_sum = 0, digamma_sum = 0;
	    double[] digamma_gam = new double[model.num_topics];
	    for(int k = 0; k < model.num_topics; k++)
	    {
	    	digamma_gam[k] = Gamma.digamma(doc.gamma[k]);
	    	gamma_sum += doc.gamma[k];
	    }
	    digamma_sum = Gamma.digamma(gamma_sum);
	    likelihood = Gamma.logGamma(model.alpha * model.num_topics) 
	    		- model.num_topics * Gamma.logGamma(model.alpha) 
	    		- Gamma.logGamma(gamma_sum);
	    
	    for(int k = 0; k < model.num_topics; k++)
	    {
	    	likelihood += (model.alpha - 1) * (digamma_gam[k] - digamma_sum) 
	    			+ Gamma.logGamma(doc.gamma[k]) 
	    			- (doc.gamma[k] - 1) * (digamma_gam[k] - digamma_sum);
	    	for(int n = 0; n < doc.length; n++)
		    {
		    	if(doc.phi[n][k] > 0)
		    	{
		    		likelihood += doc.counts[n] * (doc.phi[n][k] * 
		    				((digamma_gam[k] - digamma_sum) - 
		    				Math.log(doc.phi[n][k]) +
		    				model.log_prob_w[k][doc.ids[n]]));
		    	}
		    }
	    }
	    
	    
	    return likelihood;
	}
	
	
	

	public void save_gamma(Model model, File file)
	{
		DecimalFormat df = new DecimalFormat("#.##");
		StringBuilder sb_gamma = new StringBuilder();  //Save gamma for each EM iteration
        for(int d = 0; d < corpus.num_docs; d++)
        {
        	sb_gamma.append(corpus.docs[d].doc_name);
        	for(int k = 0; k < model.num_topics; k++)
        	{       		
        		sb_gamma.append("\t" + df.format(corpus.docs[d].gamma[k]));
        	}
        	sb_gamma.append("\n");
        }
        try {
			FileUtils.writeStringToFile(file, sb_gamma.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	
	/**
	 * Save top words of each topic from whole corpus using Beta matrix
	 * @param M     top M words
	 * @param file  output file
	 * @return topwords   K*V  top M probability words matrix, if it is top words, the value is 1, else 0 
	 */
	public int[][] save_top_words_corpus(int M, Model model, File file)
	{
		int K = num_topics;
		int V = corpus.num_terms;
		String[][] res = new String[M][K];
		int[][] topwords = new int[K][V];
		for(int k = 0; k < K; k++)
		{			
			double[] temp = new double[V];
			for(int v = 0; v < V; v++)
				temp[v] = model.log_prob_w[k][v];
			Tools tools = new Tools();
			ArrayIndexComparator comparator = tools.new ArrayIndexComparator(temp);
			Integer[] indexes = comparator.createIndexArray();
			Arrays.sort(indexes, comparator);
			for(int i = 0; i < M; i++)
			{
				res[i][k] = corpus.voc.idToWord.get(indexes[i]);
				topwords[k][indexes[i]] = 1;
			}
		}
		StringBuilder sb = new StringBuilder();
		for(int i = 0; i < M; i++)
		{
			for(int k = 0; k < K; k++)
			{
				sb.append(String.format("%-15s" , res[i][k]));
			}
			sb.append("\n");
			
		}
		try {
			FileUtils.writeStringToFile(file, sb.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}
		return topwords;
	}
	
	/**
	 * This method save parameters in GTRF model.
	 * @param corpus
	 * @param filename
	 */
	public void save_doc_para(File file)
	{
		DecimalFormat df = new DecimalFormat("#.##");
		StringBuilder sb = new StringBuilder();  //Save parameters in each document for each EM iteration
		sb.append("doc_name");    		
		sb.append("\t" + "zeta1");
		sb.append("\t" + "zeta2");
		sb.append("\t" + "num_e");
		sb.append("\t" + "exp_ec");
		sb.append("\t" + "exp_theta_square");
		sb.append("\n");
        for(int d = 0; d < corpus.num_docs; d++)
        {
        	sb.append(corpus.docs[d].doc_name);    		
    		sb.append("\t" + df.format(corpus.docs[d].zeta1));
    		sb.append("\t" + df.format(corpus.docs[d].zeta2));
    		sb.append("\t" + df.format(corpus.docs[d].num_e));
    		sb.append("\t" + df.format(corpus.docs[d].exp_ec));
    		sb.append("\t" + df.format(corpus.docs[d].exp_theta_square));
        	sb.append("\n");
        }
        try {
			FileUtils.writeStringToFile(file, sb.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void save_doc_para2(File file)
	{
		DecimalFormat df = new DecimalFormat("#.##");
		StringBuilder sb = new StringBuilder();  //Save parameters in each document for each EM iteration
		sb.append("doc_name");    		
		sb.append("\t" + "zeta1");
		sb.append("\t" + "zeta2");
		sb.append("\t" + "num_e");
		sb.append("\t" + "exp_ec");
		sb.append("\t" + "num_e2");
		sb.append("\t" + "exp_ec2");
		sb.append("\t" + "exp_theta_square");
		sb.append("\n");
        for(int d = 0; d < corpus.num_docs; d++)
        {
        	sb.append(corpus.docs[d].doc_name);    		
    		sb.append("\t" + df.format(corpus.docs[d].zeta1));
    		sb.append("\t" + df.format(corpus.docs[d].zeta2));
    		sb.append("\t" + df.format(corpus.docs[d].num_e));
    		sb.append("\t" + df.format(corpus.docs[d].exp_ec));
    		sb.append("\t" + df.format(corpus.docs[d].num_e2));
    		sb.append("\t" + df.format(corpus.docs[d].exp_ec2));
    		sb.append("\t" + df.format(corpus.docs[d].exp_theta_square));
        	sb.append("\n");
        }
        try {
			FileUtils.writeStringToFile(file, sb.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void computePerplexity_e_theta(Model model)
	{
		System.out.println("========evaluate========");
		double perplex = 0;
		int N = 0;
		StringBuilder sb = new StringBuilder();
		for(int m = 0; m < corpus.docs_test.length; m++)
		{
			Document doc = corpus.docs_test[m];
			//Initialize gamma and phi to zero for each document
			corpus.docs_test[m].gamma = new double[model.num_topics];
			corpus.docs_test[m].phi = new double[corpus.maxLength()][num_topics];
			lda_inference(doc, model);
			double[] theta = new double[num_topics];
			double theta_sum = 0;
			double gamma_sum = 0;
			for(int k = 0; k < num_topics; k++)
			{
				theta[k] = Gamma.digamma(doc.gamma[k]);
				gamma_sum += doc.gamma[k];
			}
			double diggamma_sumgamma =  Gamma.digamma(gamma_sum);
			for(int k = 0; k < num_topics; k++)
			{
				theta[k] -= diggamma_sumgamma;
				theta[k] = Math.exp(theta[k]);
			}
//			for(int k = 0; k < num_topics; k++)
//			{
//				theta[k] = doc.gamma[k];
//				theta_sum += theta[k];
//			}
//			for(int k = 0; k < num_topics; k++)
//			{
//				theta[k] = theta[k]/theta_sum;
//			}
			double log_p_w = 0;
			for(int n = 0; n < doc.length; n++)
			{
				double betaTtheta = 0;
				for(int k = 0; k < num_topics; k++)
				{
					betaTtheta += Math.exp(model.log_prob_w[k][doc.ids[n]])*theta[k];
				}
				log_p_w += doc.counts[n]*Math.log(betaTtheta);
				
			}
			N += doc.total;
			perplex += log_p_w;
		}
		perplex = Math.exp(-(perplex/N));
		perplex = Math.floor(perplex);
		System.out.println(perplex);
		sb.append("Perplexity: " + perplex);
		try {
			File eval = new File(path_res, "eval"); 
			File all_eval = new File(path, "eval");
			String s = path_res + " : " + perplex + "\n";
			FileUtils.writeStringToFile(eval, sb.toString());
			FileUtils.writeStringToFile(all_eval, s, true);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void computePerplexity_e_theta(Model model, int[][] topwords)
	{
		System.out.println("========evaluate========");
		double perplex = 0;
		int N = 0;
		StringBuilder sb = new StringBuilder();
		for(int m = 0; m < corpus.docs_test.length; m++)
		{
			Document doc = corpus.docs_test[m];
			//Initialize gamma and phi to zero for each document
			corpus.docs_test[m].gamma = new double[model.num_topics];
			corpus.docs_test[m].phi = new double[corpus.maxLength()][num_topics];
			lda_inference(doc, model);
			double[] theta = new double[num_topics];
			double theta_sum = 0;
			for(int k = 0; k < num_topics; k++)
			{
				theta[k] = doc.gamma[k];
				theta_sum += theta[k];
			}
			for(int k = 0; k < num_topics; k++)
			{
				theta[k] = theta[k]/theta_sum;
			}
			double log_p_w = 0;
			for(int n = 0; n < doc.length; n++)
			{
				boolean flag = false;
				for(int k = 0; k < num_topics; k++)
				{
					if(topwords[k][doc.ids[n]] == 1)
					{
						flag = true;
						break;
					}
				}
				if(flag == true)
				{
					double betaTtheta = 0;
					for(int k = 0; k < num_topics; k++)
					{					
						betaTtheta += Math.exp(model.log_prob_w[k][doc.ids[n]])*theta[k];
					}
					log_p_w += doc.counts[n]*Math.log(betaTtheta);
					N += doc.counts[n];
				}
				
			}
			perplex += log_p_w;
		}
		perplex = Math.exp(-(perplex/N));
		perplex = Math.floor(perplex);
		System.out.println(perplex);
		sb.append("Perplexity: " + perplex);
		try {
			File eval = new File(path_res, "eval"); 
			File all_eval = new File(path, "eval");
			String s = path_res + " : " + perplex + "\n";
			FileUtils.writeStringToFile(eval, sb.toString());
			FileUtils.writeStringToFile(all_eval, s, true);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	
	
	/**
	 * Compute perplexity only using top M words for each topic
	 * Compute perplexity using alpha, beta, Beta(topic-words) learning from training set
	 * Compute theta through gibbs sampling in test set.
	 * @param model 
	 * @param topwords   K*V  top M probability words matrix getting from save_top_words_corpus
	 * 							if it is top words, the value is 1, else 0 
	 */
	public void computePerplexity_gibbs(Model model, int[][] topwords)
	{
		System.out.println("========evaluate========");
		double perplex = 0;
		int N = 0;
		StringBuilder sb = new StringBuilder();
		GibbsSampling gibbs = new GibbsSampling(path, path_res, num_topics, corpus, 1, alpha, beta);
		gibbs.run_gibbs();
		double[][] theta = gibbs.theta;
		for(int m = 0; m < corpus.docs_test.length; m++)
		{
			Document doc = corpus.docs_test[m];
			double log_p_w = 0;
			for(int n = 0; n < doc.length; n++)
			{
				boolean flag = false;
				for(int k = 0; k < num_topics; k++)
				{
					if(topwords[k][doc.ids[n]] == 1)
					{
						flag = true;
						break;
					}
				}
				if(flag == true)
				{
					double betaTtheta = 0;
					for(int k = 0; k < num_topics; k++)
					{					
						betaTtheta += Math.exp(model.log_prob_w[k][doc.ids[n]])*theta[m][k];
					}
					log_p_w += doc.counts[n]*Math.log(betaTtheta);
					N += doc.counts[n];
				}
				
			}
			perplex += log_p_w;
		}
		perplex = Math.exp(-(perplex/N));
		perplex = Math.floor(perplex);
		System.out.println(perplex);
		System.out.println(theta[0][0]);
		sb.append("Perplexity: " + perplex);
		try {
			File eval = new File(path_res, "eval"); 
			File all_eval = new File(path, "eval");
			String s = path_res + " : " + perplex + "\n";
			FileUtils.writeStringToFile(eval, sb.toString());
			FileUtils.writeStringToFile(all_eval, s, true);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	
	/**
	 * Compute complexity using all words in test set
	 * Compute theta through gibbs sampling in test set.
	 * @param model
	 */
	public void computePerplexity_gibbs(Model model)
	{
		System.out.println("========evaluate========");
		double perplex = 0;
		int N = 0;
		StringBuilder sb = new StringBuilder();
		GibbsSampling gibbs = new GibbsSampling(path, path_res, num_topics, corpus, 1, alpha, beta, 10, 500, 1000);
		gibbs.run_gibbs();
		double[][] theta = gibbs.theta;
		for(int m = 0; m < corpus.docs_test.length; m++)
		{
			Document doc = corpus.docs_test[m];
			double log_p_w = 0;
			for(int n = 0; n < doc.length; n++)
			{
				double betaTtheta = 0;
				for(int k = 0; k < num_topics; k++)
				{
					betaTtheta += Math.exp(model.log_prob_w[k][doc.ids[n]])*theta[m][k];
				}
				log_p_w += doc.counts[n]*Math.log(betaTtheta);
				
			}
			N += doc.total;
			perplex += log_p_w;
		}
		perplex = Math.exp(-(perplex/N));
		perplex = Math.floor(perplex);
		System.out.println(perplex);
		System.out.println(theta[0][0]);
		sb.append("Perplexity: " + perplex);
		try {
			File eval = new File(path_res, "eval"); 
			File all_eval = new File(path, "eval");
			String s = path_res + " : " + perplex + "\n";
			FileUtils.writeStringToFile(eval, sb.toString());
			FileUtils.writeStringToFile(all_eval, s, true);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void pred_dist(Model model, double percentage)
	{
		System.out.println("========evaluate predictive distribution========");
		double sum_pred = 0;
		StringBuilder sb = new StringBuilder();
		for(int m = 0; m < corpus.docs_test.length; m++)
		{
			Document doc = corpus.docs_test[m];
			double[] theta = new double[num_topics];
			//Initialize gamma and phi to zero for each document
			doc.gamma = new double[model.num_topics];
			doc.phi = new double[corpus.maxLength()][num_topics];
			int num_words_train = (int) Math.round(doc.length*percentage);
			lda_inference(doc, model, num_words_train);
			double theta_sum = 0;
			double gamma_sum = 0;
			for(int k = 0; k < num_topics; k++)
			{
				theta[k] = Gamma.digamma(doc.gamma[k]);
				gamma_sum += doc.gamma[k];
			}
			double diggamma_sumgamma =  Gamma.digamma(gamma_sum);
			for(int k = 0; k < num_topics; k++)
			{
				theta[k] -= diggamma_sumgamma;
			}
//			for(int k = 0; k < num_topics; k++)
//			{
//				theta[k] = Math.log(doc.gamma[k]);
//				if (k > 0)
//                    theta_sum = Tools.log_sum(theta_sum, theta[k]);
//                else
//                	theta_sum = theta[k];
//			}
			double pred = 0;
			for(int k = 0; k < num_topics; k++)
			{
//				theta[k] = theta[k] - theta_sum;
				double beta_k_sum = 0;
				for(int n = num_words_train; n < doc.length; n++)
				{
					if(n == num_words_train)
						beta_k_sum = model.log_prob_w[k][doc.ids[n]];
					else
						beta_k_sum = Tools.log_sum(beta_k_sum, model.log_prob_w[k][doc.ids[n]]);
				}
				double beta_k_mean = beta_k_sum - Math.log(doc.length - num_words_train);
				if(k == 0)
					pred = theta[k] + beta_k_mean;
				else
					pred = Tools.log_sum(pred, theta[k] + beta_k_mean);
			}
			sum_pred += pred;
		}
		double mean_pred = sum_pred/corpus.docs_test.length;
		System.out.println(mean_pred);
		sb.append("Pred dist: " + mean_pred);
		try {
			File eval = new File(path_res, "eval_pred"); 
			File all_eval = new File(path, "eval_pred");
			String s = path_res + " : " + mean_pred + "\n";
			FileUtils.writeStringToFile(eval, sb.toString());
			FileUtils.writeStringToFile(all_eval, s, true);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public double computNMI() {
		// double res = 0;

		ArrayList<ArrayList<Integer>> textLabel = new ArrayList<ArrayList<Integer>>();

		ArrayList<Integer> labelId = new ArrayList<Integer>();

		for (int d = 0; d < corpus.docs.length; d++) {
			int id = corpus.docs[d].category;

			if (labelId.contains(id)) {
				int index = labelId.indexOf(id);

				textLabel.get(index).add(d);
			} else {
				ArrayList<Integer> subLabel = new ArrayList<Integer>();
				subLabel.add(d);
				labelId.add(id);
				textLabel.add(subLabel);
			}
		}

		ArrayList<ArrayList<Integer>> clusterLabel = new ArrayList<ArrayList<Integer>>();

		ArrayList<Integer> clusterlId = new ArrayList<Integer>();

		for (int d = 0; d < corpus.docs.length; d++) {
			int id = corpus.docs[d].cluster;

			if (clusterlId.contains(id)) {
				int index = clusterlId.indexOf(id);

				clusterLabel.get(index).add(d);
			} else {
				ArrayList<Integer> subLabel = new ArrayList<Integer>();
				subLabel.add(d);
				clusterlId.add(id);
				clusterLabel.add(subLabel);
			}
		}
    	
    	System.out.println(" the cluster number : " + clusterLabel.size());
    	
    	double comRes = 0;
    	
    	for(int i=0; i<textLabel.size(); i++)
    	{
    		for(int j=0; j<clusterLabel.size(); j++)
    		{
    			int common = commonArray(textLabel.get(i),clusterLabel.get(j));
    			
    			if(common!=0)
    				comRes += (double)common*Math.log((double)corpus.docs.length*common/(textLabel.get(i).size()*clusterLabel.get(j).size()));
    		}	
    	}
    	
    	double comL = 0;
    	for(int i=0; i<textLabel.size(); i++)
    	{
    		comL += (double)textLabel.get(i).size()*Math.log((double)textLabel.get(i).size()/corpus.docs.length);
    	}
    	
    	double comC = 0;
    	for(int j=0; j<clusterLabel.size(); j++)
    		comC += (double)clusterLabel.get(j).size()*Math.log((double)clusterLabel.get(j).size()/corpus.docs.length);
    	
    	//System.out.println(comRes + " " + comL + " "+ comC);
    	
    	comRes /= Math.sqrt(comL*comC);
    	/*for(int i=0; i<clusterLabel.size(); i++)
    	{
    		System.out.println(i + " " +clusterLabel.get(i).toString());
    	}*/
    	
    	return comRes;
    }
	
	/**
	 * 
	 * @param arr1
	 * @param arr2
	 * @return number of common elements in arr1 and arr2
	 */
	public int commonArray(ArrayList<Integer> arr1, ArrayList<Integer> arr2)
    {
    	int count = 0;
    	for(int i=0; i<arr1.size(); i++)
    		if(arr2.contains(arr1.get(i)))
    			count++;
    	
    	return count;
    }
	
	/**
	 * E step for evaluation method predictive distribution.
	 * @param doc
	 * @param model
	 * @param num_words_train  number of words used for training   w_obs
	 * @return
	 */
	public double lda_inference(Document doc, Model model, int num_words_train)
	{		
	    double likelihood = 0, likelihood_old = 0;
	    double[] digamma_gam = new double[model.num_topics];
	    
	    // compute posterior dirichlet
	    
	    //initialize varitional parameters gamma and phi
	    for (int k = 0; k < model.num_topics; k++)
	    {
	        doc.gamma[k] = model.alpha + (doc.total/((double) model.num_topics));
	        //compute digamma gamma for later use
	        digamma_gam[k] = Gamma.digamma(doc.gamma[k]);
	        for (int n = 0; n < doc.length; n++)
	            doc.phi[n][k] = 1.0/model.num_topics;
	    }
	    
	    double converged = 1;
	    int var_iter = 0;
	    double[] oldphi = new double[model.num_topics];  //????
	    while (converged > VAR_CONVERGED && var_iter < VAR_MAX_ITER)
	    {
	    	for(int k = 0; k < num_topics; k++)
	    	{
	        	doc.gamma[k] = 0;
	    	}
	    	var_iter++;
//	    	System.out.println("var_iter: " + var_iter);
	    	for(int n = 0; n < num_words_train; n++)
	    	{
	    		double phisum = 0;
	    		for(int k = 0; k < model.num_topics; k++)
	    		{
	    			oldphi[k] = doc.phi[n][k];
	    			//phi = beta * exp(digamma(gamma)) -> log phi = log (beta) + digamma(gamma)
	    			doc.phi[n][k] = model.log_prob_w[k][doc.ids[n]] + digamma_gam[k];
	    			if (k > 0)
	                    phisum = Tools.log_sum(phisum, doc.phi[n][k]);
	                else
	                    phisum = doc.phi[n][k]; // note, phi is in log space
	    		}	    		
	    		for (int k = 0; k < model.num_topics; k++)
	            {
	    			//Normalize phi, exp(log phi - log phisum) = phi/phisum
	                doc.phi[n][k] = Math.exp(doc.phi[n][k] - phisum);
//	                doc.gamma[k] += doc.counts[n]*(doc.phi[n][k] - oldphi[k]);
	                doc.gamma[k] += doc.counts[n]*doc.phi[n][k];
//	                digamma_gam[k] = doc.gamma[k] > 0? Gamma.digamma(doc.gamma[k]):Gamma.digamma(0.1);
	            }

	    	}
	    	for(int k = 0; k < num_topics; k++)
	    	{
	    		doc.gamma[k] += model.alpha;
	    		digamma_gam[k] = Gamma.digamma(doc.gamma[k]);
	    	}
	    	likelihood = compute_likelihood(doc, model, num_words_train);
//		    System.out.println("likelihood: " + likelihood);		    
		    converged = (likelihood_old - likelihood) / likelihood_old;
//		    System.out.println(converged);
	        likelihood_old = likelihood;
	    }
	    
	    return likelihood;
	}
	
	/**
	 * compute likelihood of num_words_train words in a document, in the evaluation method predictive distribution
	 * @param doc
	 * @param model
	 * @param num_words_train
	 * @return
	 */
	public double compute_likelihood(Document doc, Model model, int num_words_train)
	{
		double likelihood = 0, gamma_sum = 0, digamma_sum = 0;
	    double[] digamma_gam = new double[model.num_topics];
	    for(int k = 0; k < model.num_topics; k++)
	    {
	    	digamma_gam[k] = Gamma.digamma(doc.gamma[k]);
	    	gamma_sum += doc.gamma[k];
	    }
	    digamma_sum = Gamma.digamma(gamma_sum);
	    likelihood = Gamma.logGamma(model.alpha * model.num_topics) 
	    		- model.num_topics * Gamma.logGamma(model.alpha) 
	    		- Gamma.logGamma(gamma_sum);
	    
	    for(int k = 0; k < model.num_topics; k++)
	    {
	    	likelihood += (model.alpha - 1) * (digamma_gam[k] - digamma_sum) 
	    			+ Gamma.logGamma(doc.gamma[k]) 
	    			- (doc.gamma[k] - 1) * (digamma_gam[k] - digamma_sum);
	    	for(int n = 0; n < num_words_train; n++)
		    {
		    	if(doc.phi[n][k] > 0)
		    	{
		    		likelihood += doc.counts[n] * (doc.phi[n][k] * 
		    				((digamma_gam[k] - digamma_sum) - 
		    				Math.log(doc.phi[n][k]) +
		    				model.log_prob_w[k][doc.ids[n]]));
		    	}
		    }
	    }	    
	    return likelihood;
	}

}
