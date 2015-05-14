package sgtrf;

import java.util.List;

import org.apache.commons.math3.special.Gamma;

import base_model.Corpus;
import base_model.Document;
import base_model.EM;
import base_model.Model;
import base_model.Tools;

/**
 * This class is similar to EM_m, except using word similarity 
 */

public class EM_s extends EM {
	public double lambda2;
	public double lambda4;
	public double[][] sim = null;

	public EM_s(String path, String path_res, int num_topics, Corpus corpus, double beta, double lambda2, double lambda4, 
			double[][] sim) {
		super(path, path_res, num_topics, corpus, beta);
		this.lambda2 = lambda2;
		this.lambda4 = lambda4;
		this.sim = sim;
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
		    double[][] sumadj = new double[doc.length][model.num_topics];
		    double[][] sumadj2 = new double[doc.length][model.num_topics];
		    double exp_ec = 0;  //expectation of coherent edges;
		    double exp_ec2 = 0;  //expectation of coherent edges with distance 2;

	    	for(int n = 0; n < doc.length; n++)
	    	{
	    		for(int k = 0; k < model.num_topics; k++)
	    		{
	    			oldphi[n][k] = doc.phi[n][k];
	    			if(doc.adj.containsKey(doc.ids[n]))
	    			{
	    				List<Integer> list = doc.adj.get(doc.ids[n]);
//	    				System.out.println(list.toString());
		    			for(int adj_id = 0; adj_id < list.size(); adj_id++)
		    			{
		    				double similarity = (sim[doc.ids[n]][list.get(adj_id)] + 1)/2.0;
							sumadj[n][k] += oldphi[doc.idToIndex.get(list.get(adj_id))][k]*similarity;
							exp_ec += oldphi[n][k] * oldphi[doc.idToIndex.get(list.get(adj_id))][k]*similarity;
		    			}
	    			}
	    			if(doc.adj2.containsKey(doc.ids[n]))
	    			{
	    				List<Integer> list = doc.adj2.get(doc.ids[n]);
//	    				System.out.println(list.toString());
		    			for(int adj_id = 0; adj_id < list.size(); adj_id++)
		    			{
		    				double similarity = (sim[doc.ids[n]][list.get(adj_id)] + 1)/2.0;
							sumadj2[n][k] += doc.counts[n] * oldphi[doc.idToIndex.get(list.get(adj_id))][k]*similarity;
							exp_ec2 += oldphi[n][k] * oldphi[doc.idToIndex.get(list.get(adj_id))][k]*similarity;
		    			}
	    			}
	    		}
	    	}
	    	doc.exp_ec = exp_ec/2; //divided by 2 because we count each node twice
	    	doc.exp_ec2 = exp_ec2/2;
	    	
	    	double exp_theta_square = 0;
	    	double sum_gamma = 0;
	    	for(int k = 0; k < num_topics; k++)
	    	{
	    		exp_theta_square += doc.gamma[k]*(doc.gamma[k] + 1);
	    		sum_gamma += doc.gamma[k];
	    	}
	    	doc.exp_theta_square = exp_theta_square/(sum_gamma*(sum_gamma + 1));
	    	
	    	doc.zeta1 = (1 - lambda2)*doc.exp_ec + lambda2 * doc.num_e * doc.exp_theta_square + 
	    			(1 - lambda4)*doc.exp_ec2 + lambda4 * doc.num_e2 * doc.exp_theta_square;
	    	doc.zeta2 = (doc.num_e + doc.num_e2) * doc.exp_theta_square;
	    	
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
	    					((1 - lambda2)/doc.zeta1)*sumadj[n][k] +
	    					((1 - lambda4)/doc.zeta1)*sumadj2[n][k];
	    			if (k > 0)
	                    phisum = Tools.log_sum(phisum, doc.phi[n][k]);
	                else
	                    phisum = doc.phi[n][k]; // note, phi is in log space
	    		}	    		
	    		for (int k = 0; k < model.num_topics; k++)
	            {
	    			//Normalize phi, exp(log phi - log phisum) = phi/phisum
	                doc.phi[n][k] = Math.exp(doc.phi[n][k] - phisum);
//	                doc.gamma[k] += doc.counts[n]*(doc.phi[n][k] - oldphi[n][k]);
	                doc.gamma[k] += doc.counts[n]*doc.phi[n][k];
//	                digamma_gam[k] = doc.gamma[k] > 0? Gamma.digamma(doc.gamma[k]):Gamma.digamma(0.1);
	            }

	    	}
    		for (int k = 0; k < model.num_topics; k++)
    		{
    			doc.gamma[k] += model.alpha;
    			doc.gamma[k] = updataGamma(doc.gamma[k], old_gamma[k], sum_gamma, doc.zeta1, doc.zeta2, doc.num_e, doc.num_e2, model);
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
	
	public double updataGamma(double lda_gamma, double old_gamma, double sum_gamma, double zeta1, double zeta2, int num_e, int num_e2, Model model)
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
					((zeta1*(num_e + num_e2) - zeta2*(num_e*lambda2 + num_e2*lambda4))/(zeta1 * zeta2)) *
					(2*x0*Math.pow(sum_gamma,2)+Math.pow(sum_gamma, 2)+sum_gamma - 2*Math.pow(x0, 2)*sum_gamma - Math.pow(x0,2) - x0 )/
					(Math.pow(sum_gamma, 2)*Math.pow(sum_gamma + 1, 2));
			if(Math.abs(f) < 0.001)
				break;
			double df =(Gamma.trigamma(sum_gamma) - Gamma.trigamma(x0)) + (lda_gamma - x0)*(Tools.tetragamma(x0) - Tools.tetragamma(sum_gamma)) - 
					((zeta1*(num_e + num_e2) - zeta2*(num_e*lambda2 + num_e2*lambda4))/(zeta1 * zeta2))*
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
		likelihood += ((1 - lambda2)/doc.zeta1)*doc.exp_ec + ((1 - lambda4)/doc.zeta1)*doc.exp_ec2 - 
	    		((doc.zeta1*(doc.num_e + doc.num_e2) - doc.zeta2*(doc.num_e*lambda2 + doc.num_e2*lambda4))/(doc.zeta1 * doc.zeta2))*doc.exp_theta_square + 
	    		Math.log(doc.zeta1) - Math.log(doc.zeta2);
		return likelihood;
	}
	
	@Override
	public double compute_likelihood(Document doc, Model model, int num_words_train) {
		// TODO Auto-generated method stub
		double likelihood = super.compute_likelihood(doc, model, num_words_train);
		likelihood += ((1 - lambda2)/doc.zeta1)*doc.exp_ec + ((1 - lambda4)/doc.zeta1)*doc.exp_ec2 - 
	    		((doc.zeta1*(doc.num_e + doc.num_e2) - doc.zeta2*(doc.num_e*lambda2 + doc.num_e2*lambda4))/(doc.zeta1 * doc.zeta2))*doc.exp_theta_square + 
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
		    double[][] sumadj = new double[doc.length][model.num_topics];
		    double[][] sumadj2 = new double[doc.length][model.num_topics];
		    double exp_ec = 0;  //expectation of coherent edges;
		    double exp_ec2 = 0;  //expectation of coherent edges with distance 2;

	    	for(int n = 0; n < num_words_train; n++)
	    	{
	    		for(int k = 0; k < model.num_topics; k++)
	    		{
	    			oldphi[n][k] = doc.phi[n][k];
	    			if(doc.adj.containsKey(doc.ids[n]))
	    			{
	    				List<Integer> list = doc.adj.get(doc.ids[n]);
//	    				System.out.println(list.toString());
		    			for(int adj_id = 0; adj_id < list.size(); adj_id++)
		    			{
		    				double similarity = (sim[doc.ids[n]][list.get(adj_id)] + 1)/2.0;
							sumadj[n][k] += oldphi[doc.idToIndex.get(list.get(adj_id))][k]*similarity;
							exp_ec += oldphi[n][k] * oldphi[doc.idToIndex.get(list.get(adj_id))][k]*similarity;
		    			}
	    			}
	    			if(doc.adj2.containsKey(doc.ids[n]))
	    			{
	    				List<Integer> list = doc.adj2.get(doc.ids[n]);
//	    				System.out.println(list.toString());
		    			for(int adj_id = 0; adj_id < list.size(); adj_id++)
		    			{
		    				double similarity = (sim[doc.ids[n]][list.get(adj_id)] + 1)/2.0;
							sumadj2[n][k] += doc.counts[n] * oldphi[doc.idToIndex.get(list.get(adj_id))][k]*similarity;
							exp_ec2 += oldphi[n][k] * oldphi[doc.idToIndex.get(list.get(adj_id))][k]*similarity;
		    			}
	    			}
	    		}
	    	}
	    	doc.exp_ec = exp_ec/2; //divided by 2 because we count each node twice
	    	doc.exp_ec2 = exp_ec2/2;
	    	
	    	double exp_theta_square = 0;
	    	double sum_gamma = 0;
	    	for(int k = 0; k < num_topics; k++)
	    	{
	    		exp_theta_square += doc.gamma[k]*(doc.gamma[k] + 1);
	    		sum_gamma += doc.gamma[k];
	    	}
	    	doc.exp_theta_square = exp_theta_square/(sum_gamma*(sum_gamma + 1));
	    	
	    	doc.zeta1 = (1 - lambda2)*doc.exp_ec + lambda2 * doc.num_e * doc.exp_theta_square + 
	    			(1 - lambda4)*doc.exp_ec2 + lambda4 * doc.num_e2 * doc.exp_theta_square;
	    	doc.zeta2 = (doc.num_e + doc.num_e2) * doc.exp_theta_square;
	    	
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
	    					((1 - lambda2)/doc.zeta1)*sumadj[n][k] +
	    					((1 - lambda4)/doc.zeta1)*sumadj2[n][k];
	    			if (k > 0)
	                    phisum = Tools.log_sum(phisum, doc.phi[n][k]);
	                else
	                    phisum = doc.phi[n][k]; // note, phi is in log space
	    		}	    		
	    		for (int k = 0; k < model.num_topics; k++)
	            {
	    			//Normalize phi, exp(log phi - log phisum) = phi/phisum
	                doc.phi[n][k] = Math.exp(doc.phi[n][k] - phisum);
//	                doc.gamma[k] += doc.counts[n]*(doc.phi[n][k] - oldphi[n][k]);
	                doc.gamma[k] += doc.counts[n]*doc.phi[n][k];
//	                digamma_gam[k] = doc.gamma[k] > 0? Gamma.digamma(doc.gamma[k]):Gamma.digamma(0.1);
	            }

	    	}
    		for (int k = 0; k < model.num_topics; k++)
    		{
    			doc.gamma[k] += model.alpha;
    			doc.gamma[k] = updataGamma(doc.gamma[k], old_gamma[k], sum_gamma, doc.zeta1, doc.zeta2, doc.num_e, doc.num_e2, model);
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
