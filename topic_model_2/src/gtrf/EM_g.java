package gtrf;

import java.util.List;

import org.apache.commons.math3.special.Gamma;

import base_model.Corpus;
import base_model.Document;
import base_model.EM;
import base_model.Model;
import base_model.Tools;


public class EM_g extends EM{

	public double lambda2;
	
	public EM_g(String path, String path_res, int num_topics, Corpus corpus, double lambda2,
			int vAR_MAX_ITER, double vAR_CONVERGED, int eM_MAX_ITER,
			double eM_CONVERGED, double alpha, double beta) {
		super(path, path_res, num_topics, corpus, vAR_MAX_ITER, vAR_CONVERGED, eM_MAX_ITER,
				eM_CONVERGED, alpha, beta);
		this.lambda2 = lambda2;
	}

	public EM_g(String path, String path_res, int num_topics, Corpus corpus, double beta, double lambda2) {
		super(path, path_res, num_topics, corpus, beta);
		this.lambda2 = lambda2;
	}

	@Override
	public double lda_inference(Document doc, Model model) {
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
	    double[][] oldphi = new double[doc.length][model.num_topics];	    
	    while (converged > VAR_CONVERGED && var_iter < VAR_MAX_ITER)
	    {
	    	var_iter++;
	    	
	    	//Store old phi
	    	//sum over phi of all adj nodes of current node
		    double[][] sumadj = new double[doc.length][model.num_topics]; //sum of phi of words which are adjacent to current node
		    double exp_ec = 0;  //expectation of coherent edges;
	    	for(int n = 0; n < doc.length; n++)
	    	{
	    		for(int k = 0; k < model.num_topics; k++)
	    		{
	    			oldphi[n][k] = doc.phi[n][k];
	    			if(doc.adj.containsKey(doc.ids[n]))
	    			{
	    				List<Integer> list = doc.adj.get(doc.ids[n]);
		    			for(int adj_id = 0; adj_id < list.size(); adj_id++)
		    			{
//		    				System.out.println(doc.doc_name);
							sumadj[n][k] += oldphi[doc.idToIndex.get(list.get(adj_id))][k];
							exp_ec += oldphi[n][k] * oldphi[doc.idToIndex.get(list.get(adj_id))][k];
							
		    			}
	    			}
	    		}
	    	}
	    	doc.exp_ec = exp_ec/2; //divided by 2 because we count each node twice
	    	
	    	double exp_theta_square = 0;
	    	double sum_gamma = 0;
	    	for(int k = 0; k < num_topics; k++)
	    	{
	    		exp_theta_square += doc.gamma[k]*(doc.gamma[k] + 1);
	    		sum_gamma += doc.gamma[k];
	    	}
	    	doc.exp_theta_square = exp_theta_square/(sum_gamma*(sum_gamma + 1));
	    	
	    	doc.zeta1 = (1 - lambda2)*doc.exp_ec + lambda2 * doc.num_e * doc.exp_theta_square;
	    	doc.zeta2 = doc.num_e * doc.exp_theta_square;
	    	
	    	//Set gamma[k] to zero to update gamma
	    	double[] old_gamma = new double[num_topics];
	    	for(int k = 0; k < num_topics; k++)
	    	{
	    		old_gamma[k] = doc.gamma[k];
	        	doc.gamma[k] = 0;
	    	}
	    	for(int n = 0; n < doc.length; n++)
	    	{
	    		double phisum = 0;
	    		for(int k = 0; k < model.num_topics; k++)
	    		{	    			
	    			//phi = beta * exp(digamma(gamma) + (1-lambda2)/zeta1 * sum(phi(m, i)))  m is adj of n 
	    			//-> log phi = log (beta) + digamma(gamma) + (1-lambda2)/zeta1 * sum(phi(m, i))
	    			doc.phi[n][k] = model.log_prob_w[k][doc.ids[n]] + digamma_gam[k] + 
	    					((1 - lambda2)/doc.zeta1)*sumadj[n][k];
	    			if (k > 0)
	                    phisum = Tools.log_sum(phisum, doc.phi[n][k]);
	                else
	                    phisum = doc.phi[n][k]; // note, phi is in log space
	    		}	    		
	    		for (int k = 0; k < model.num_topics; k++)
	            {
	    			//Normalize phi, exp(log phi - log phisum) = phi/phisum
	                doc.phi[n][k] = Math.exp(doc.phi[n][k] - phisum);
//	                doc.gamma[k] += doc.counts[n]*(doc.phi[n][k]-oldphi[n][k]);
	                doc.gamma[k] += doc.counts[n]*doc.phi[n][k];
//	                digamma_gam[k] = doc.gamma[k] > 0? Gamma.digamma(doc.gamma[k]):Gamma.digamma(0.1);
	            }

	    	}
	    	for(int k = 0; k < num_topics; k++)
	    	{
	    		doc.gamma[k] += model.alpha;
	    		doc.gamma[k] = updataGamma(doc.gamma[k], old_gamma[k], sum_gamma, doc.zeta1, doc.zeta2, doc.num_e, model);
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
	
	public double updataGamma(double lda_gamma, double old_gamma, double sum_gamma, double zeta1, double zeta2, int num_e, Model model)
	{
		double e = 1e-4;
		int iters = 50;
		double x0 = lda_gamma;
		double x1 = 0;
		int iter = 0;
		double converged = 1;
		sum_gamma = sum_gamma - old_gamma + lda_gamma;
		while(converged > e && iter <= iters)
		{
			iter++;
			
			double f = (lda_gamma - x0)*(Gamma.trigamma(x0) - Gamma.trigamma(sum_gamma)) - 
					(((zeta1 - zeta2*lambda2)*num_e)/(zeta1*zeta2)) *
					(2*x0*Math.pow(sum_gamma,2)+Math.pow(sum_gamma, 2)+sum_gamma - 2*Math.pow(x0, 2)*sum_gamma - Math.pow(x0,2) - x0 )/
					(Math.pow(sum_gamma, 2)*Math.pow(sum_gamma + 1, 2));
			if(Math.abs(f) < 0.001)
				break;
			double df =(Gamma.trigamma(sum_gamma) - Gamma.trigamma(x0)) + (lda_gamma - x0)*(Tools.tetragamma(x0) - Tools.tetragamma(sum_gamma)) - 
					(((zeta1 - zeta2*lambda2)*num_e)/(zeta1*zeta2))*
					((2*Math.pow(sum_gamma, 2) + 2*sum_gamma - 2*Math.pow(x0,2) - 2*x0)*Math.pow(sum_gamma, 2)*Math.pow(sum_gamma + 1, 2) - 
					(2*x0*Math.pow(sum_gamma,2)+Math.pow(sum_gamma, 2)+sum_gamma - 2*Math.pow(x0, 2)*sum_gamma - Math.pow(x0,2) - x0 ) *
					(2*sum_gamma*Math.pow(sum_gamma + 1, 2) + 2*Math.pow(sum_gamma, 2)*(sum_gamma + 1)))/
					(Math.pow(sum_gamma, 4)*Math.pow(sum_gamma + 1, 4));
			x1 = x0 - f/df;			
			converged = Math.abs((x1 - x0)/x0);
			sum_gamma = sum_gamma - x0 + x1;
			x0 = x1;			 
		}
		return x0;
	}

	@Override
	public double compute_likelihood(Document doc, Model model) {
		// TODO Auto-generated method stub
		double likelihood = super.compute_likelihood(doc, model);
		
		likelihood += ((1 - lambda2)/doc.zeta1)*doc.exp_ec - 
	    		(((doc.zeta1 - doc.zeta2 * lambda2)/(doc.zeta1 * doc.zeta2)) * doc.num_e)*doc.exp_theta_square + 
	    		Math.log(doc.zeta1) - Math.log(doc.zeta2);
		
		return likelihood;
	}
	
	@Override
	public double compute_likelihood(Document doc, Model model, int num_words_train) {
		// TODO Auto-generated method stub
		double likelihood = super.compute_likelihood(doc, model, num_words_train);
		
		likelihood += ((1 - lambda2)/doc.zeta1)*doc.exp_ec - 
	    		(((doc.zeta1 - doc.zeta2 * lambda2)/(doc.zeta1 * doc.zeta2)) * doc.num_e)*doc.exp_theta_square + 
	    		Math.log(doc.zeta1) - Math.log(doc.zeta2);
		
		return likelihood;
	}
	
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
	    double[][] oldphi = new double[doc.length][model.num_topics];	    
	    while (converged > VAR_CONVERGED && var_iter < VAR_MAX_ITER)
	    {
	    	var_iter++;
	    	
	    	//Store old phi
	    	//sum over phi of all adj nodes of current node
		    double[][] sumadj = new double[doc.length][model.num_topics]; //sum of phi of words which are adjacent to current node
		    double exp_ec = 0;  //expectation of coherent edges;
	    	for(int n = 0; n < num_words_train; n++)
	    	{
	    		for(int k = 0; k < model.num_topics; k++)
	    		{
	    			oldphi[n][k] = doc.phi[n][k];
	    			if(doc.adj.containsKey(doc.ids[n]))
	    			{
	    				List<Integer> list = doc.adj.get(doc.ids[n]);
		    			for(int adj_id = 0; adj_id < list.size(); adj_id++)
		    			{
//		    				System.out.println(doc.doc_name);
							sumadj[n][k] += oldphi[doc.idToIndex.get(list.get(adj_id))][k];
							exp_ec += oldphi[n][k] * oldphi[doc.idToIndex.get(list.get(adj_id))][k];
							
		    			}
	    			}
	    		}
	    	}
	    	doc.exp_ec = exp_ec/2; //divided by 2 because we count each node twice
	    	
	    	double exp_theta_square = 0;
	    	double sum_gamma = 0;
	    	for(int k = 0; k < num_topics; k++)
	    	{
	    		exp_theta_square += doc.gamma[k]*(doc.gamma[k] + 1);
	    		sum_gamma += doc.gamma[k];
	    	}
	    	doc.exp_theta_square = exp_theta_square/(sum_gamma*(sum_gamma + 1));
	    	
//	    	if(doc.doc_name.equals("15822"))
//		    {
//		    	System.out.println("===========exp_ec============" + doc.exp_ec);
//		    	System.out.println("===========num_e============" + doc.num_e);
//		    	System.out.println("===========exp_theta_square============" + doc.exp_theta_square);
//		    	System.out.println("===========re============" + (1 - lambda2)*doc.exp_ec + lambda2 * doc.num_e * doc.exp_theta_square);
//		    }
	    	doc.zeta1 = (1 - lambda2)*doc.exp_ec + lambda2 * doc.num_e * doc.exp_theta_square;
	    	doc.zeta2 = doc.num_e * doc.exp_theta_square;
	    	
	    	double[] old_gamma = new double[num_topics];
	    	for(int k = 0; k < num_topics; k++)
	    	{
	    		old_gamma[k] = doc.gamma[k];
	        	doc.gamma[k] = 0;
	    	}
	    	
	    	for(int n = 0; n < num_words_train; n++)
	    	{
	    		double phisum = 0;
	    		for(int k = 0; k < model.num_topics; k++)
	    		{	    			
	    			//phi = beta * exp(digamma(gamma) + (1-lambda2)/zeta1 * sum(phi(m, i)))  m is adj of n 
	    			//-> log phi = log (beta) + digamma(gamma) + (1-lambda2)/zeta1 * sum(phi(m, i))
	    			doc.phi[n][k] = model.log_prob_w[k][doc.ids[n]] + digamma_gam[k] + 
	    					((1 - lambda2)/doc.zeta1)*sumadj[n][k];
	    			if (k > 0)
	                    phisum = Tools.log_sum(phisum, doc.phi[n][k]);
	                else
	                    phisum = doc.phi[n][k]; // note, phi is in log space
	    		}	    		
	    		for (int k = 0; k < model.num_topics; k++)
	            {
	    			//Normalize phi, exp(log phi - log phisum) = phi/phisum
	                doc.phi[n][k] = Math.exp(doc.phi[n][k] - phisum);
//	                doc.gamma[k] += doc.counts[n]*(doc.phi[n][k]-oldphi[n][k]);
	                doc.gamma[k] += doc.counts[n]*doc.phi[n][k];
//	                digamma_gam[k] = doc.gamma[k] > 0? Gamma.digamma(doc.gamma[k]):Gamma.digamma(0.1);
	            }

	    	}
	    	for(int k = 0; k < num_topics; k++)
	    	{
	    		doc.gamma[k] += model.alpha;
	    		doc.gamma[k] = updataGamma(doc.gamma[k], old_gamma[k], sum_gamma, doc.zeta1, doc.zeta2, doc.num_e, model);
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

}
