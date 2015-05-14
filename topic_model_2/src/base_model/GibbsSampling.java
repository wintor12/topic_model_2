package base_model;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import org.apache.commons.io.FileUtils;
import org.apache.commons.math3.random.MersenneTwister;

import base_model.Tools.ArrayIndexComparator;

public class GibbsSampling {
	public int num_topics; // topic numbers
	public int iterations;
	public double alpha;     //doc-topic dirichlet prior parameter
	public double beta;      //topic-word dirichlet prior parameter
	public String path = ""; //running path
	public String path_res = "";
	public String path_gibbs = "";
	public Corpus corpus;
	public int type;         //0  training,  1 testing
	
	int [][] nmk;//given document m, count times of topic k. M*K
	int [][] nkt;//given topic k, count times of term t. K*V
	int [] nmkSum;//Sum for each row in nmk
	public int [] nktSum;//Sum for each row in nkt
	double [][] phi;//Parameters for topic-word distribution K*V
	double [][] theta;//Parameters for doc-topic distribution M*K
	
	int [][] old_nmk;//given document m, count times of topic k. M*K
	int [][] old_nkt;//given topic k, count times of term t. K*V
	int [] old_nmkSum;//Sum for each row in nmk
	public int[] old_nktSum;//Sum for each row in nkt

	
	int saveStep;//The number of iterations between two saving
	int beginSaveIters;//Begin save model at this iteration
	
	public int V, K, M;//vocabulary size, topic number, document number
	MersenneTwister mt = new MersenneTwister(32);
	MersenneTwister mt2 = new MersenneTwister(48);
	
	
	
	public GibbsSampling(String path, String path_res, int num_topics, Corpus corpus, int type, double alpha, double beta,
			int saveStep, int beginSaveIters, int iterations) {
		this.num_topics = num_topics;
		this.iterations = iterations;
		this.path = path;
		this.path_res = path_res;
		this.corpus = corpus;
		this.type = type;
		this.saveStep = saveStep;
		this.beginSaveIters = beginSaveIters;
		this.beta = beta;
		this.alpha = alpha;
	}

	public GibbsSampling(String path, String path_res, int num_topics, Corpus corpus, int type) {
		this.path = path;
		this.path_res = path_res;
		this.num_topics = num_topics; // topic numbers
		this.alpha = 0.1;
		this.beta = 0.1;
		this.corpus = corpus;
		this.saveStep = 10;
		this.beginSaveIters = 100;
		this.iterations = 200;
		this.type = type;
	}
	
	public GibbsSampling(String path, String path_res, int num_topics, Corpus corpus, int type, double alpha, double beta) {
		this.path = path;
		this.path_res = path_res;
		this.num_topics = num_topics; // topic numbers
		this.beta = beta;
		this.corpus = corpus;
		this.saveStep = 10;
		this.beginSaveIters = 100;
		this.iterations = 200;
		this.type = type;
		this.alpha = alpha;
	}
	
	public void run_gibbs()
	{
		System.out.println("1 Initialize the model ...");
		initializeModel();
		System.out.println("2 Learning and Saving the model ...");
		inferenceModel();
//		for(int i = 0; i < K; i++)
//		{
//			System.out.println();
//			for(int j = 0; j < 10; j++)
//			{
//				System.out.print(phi[i][j] + "\t");
////				System.out.print((nkt[i][j] + beta) + "\t");
//			}
//		}
//		System.out.println();
//		for(int i = 0; i < K; i++)
//		{
//			System.out.println(nktSum[i] + V*beta);
//			for(int j = 0; j < 10; j++)
//			{
//				System.out.print((nkt[i][j]) + "\t");
//				
//			}
//			System.out.println();
//		}
			
	}
	
	public void initializeModel() {
		// TODO Auto-generated method stub
		if(type == 0)
			M = corpus.docs.length;
		else
			M = corpus.docs_test.length;
		K = num_topics;
		V = corpus.voc.size();
		nmk = new int[M][K];
		nkt = new int[K][V];    //V is ids of vocabulary
		nmkSum = new int[M];
		nktSum = new int[K];
		phi = new double[K][V];
		theta = new double[M][K];
		
		
		for(int m = 0; m < M; m++)
		{
			Document doc;
			if(type == 0)
				doc = corpus.docs[m];
			else
				doc = corpus.docs_test[m];
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
	}
	
	
	public void inferenceModel()
	{
		Document[] docs;
		if(type == 0)
			docs = corpus.docs;	
		else
			docs = corpus.docs_test;
		if(iterations < saveStep + beginSaveIters){
			System.err.println("Error: the number of iterations should be larger than " + (saveStep + beginSaveIters));
			System.exit(0);
		}
		path_gibbs = new File(path_res, "gibbs_sampling").getAbsolutePath();

		for(int i = 0; i < iterations; i++){
			if((i >= beginSaveIters) && (((i - beginSaveIters) % saveStep) == 0)){
				//Saving the model
				System.out.println("Saving model at iteration " + i +" ... ");
				//update parameters

				updateEstimatedParameters();
				if(type == 0)
				{
					saveIteratedModel(new File(path_gibbs, i+""));
				}
			}

			//Use Gibbs Sampling to update z[][]
			for(int m = 0; m < M; m++){
				Document doc = docs[m];
				for(int n = 0; n < doc.length; n++){
					// Sample from p(z_i|z_-i, w)
					int newTopic = sampleTopicZ(m, n);
					doc.z[n] = newTopic;
				}
			}
		}
		if(type == 0)
		{
			saveIteratedModel(new File(path_res, "gibbs_final"));
			int[][] topwords = save_top_words_corpus(20, new File(path_res, "top_words_corpus_gibbs"));
//			computePerplexity(phi, topwords);
			computePerplexity(phi);
		}
	}
	
	private void updateEstimatedParameters() {
		for(int k = 0; k < K; k++){
			for(int t = 0; t < V; t++){
				phi[k][t] = (nkt[k][t] + beta) / (nktSum[k] + V * beta);
			}
		}
		
		for(int m = 0; m < M; m++){
			for(int k = 0; k < K; k++){
				theta[m][k] = (nmk[m][k] + alpha) / (nmkSum[m] + K * alpha);
			}
		}
	}
	
	public int sampleTopicZ(int m, int n) {
		
		// TODO Auto-generated method stub
		// Sample from p(z_i|z_-i, w) using Gibbs upde rule
		Document doc;
		if(type == 0)
			doc = corpus.docs[m];
		else
			doc = corpus.docs_test[m];
		int id = doc.ids[n];
		//Remove topic label for w_{m,n}
		int oldTopic = doc.z[n];
		nmk[m][oldTopic]--;
		nkt[oldTopic][id]--;
		nmkSum[m]--;
		nktSum[oldTopic]--;
		
		//Compute p(z_i = k|z_-i, w)
		double [] p = new double[K];
		for(int k = 0; k < K; k++){
			p[k] = (nkt[k][id] + beta) / (nktSum[k] + V * beta) * (nmk[m][k] + alpha) / (nmkSum[m] + K * alpha);
		}
		
		//Sample a new topic label for w_{m, n} like roulette
		//Compute cumulated probability for p
		for(int k = 1; k < K; k++){
			p[k] += p[k - 1];
		}
		double u = mt2.nextDouble() * p[K - 1];
//		double u = Math.random() * p[K - 1]; //p[] is unnormalised
		int newTopic;
		for(newTopic = 0; newTopic < K; newTopic++){
			if(u < p[newTopic]){
				break;
			}
		}
		
		//Add new topic label for w_{m, n}
		nmk[m][newTopic]++;
		nkt[newTopic][id]++;
		nmkSum[m]++;
		nktSum[newTopic]++;
		return newTopic;
	}
	
	
	private void saveIteratedModel(File file)
	{
		ArrayList<String> lines = new ArrayList<String>();
		lines.add("alpha = " + alpha);
		lines.add("beta = " + beta);
		lines.add("topicNum = " + K);
		lines.add("docNum = " + M);
		lines.add("termNum = " + V);
		lines.add("iterations = " + iterations);
		lines.add("saveStep = " + saveStep);
		lines.add("beginSaveIters = " + beginSaveIters);
		try {
			FileUtils.writeLines(file, lines);
		} catch (IOException e) {
			e.printStackTrace();
		}
		String path_phi = file.getAbsolutePath() + "_phi";
		StringBuilder sb = new StringBuilder();
		for(int k = 0; k < K; k++)
		{
			for(int v = 0; v < V; v++)
			{
				sb.append(phi[k][v] + "\t");
			}
			sb.append("\n");
		}
		try {
			FileUtils.writeStringToFile(new File(path_phi), sb.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Save top words of each topic from whole corpus using beta
	 * @param M     top M words
	 * @param file  output file
	 */
	public int[][] save_top_words_corpus(int M, File file)
	{
		int K = num_topics;
		int V = corpus.num_terms;
		String[][] res = new String[M][K];
		int[][] topwords = new int[K][V];
		for(int k = 0; k < K; k++)
		{			
			double[] temp = new double[V];
			for(int v = 0; v < V; v++)
				temp[v] = phi[k][v];
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
	
	
	public void computePerplexity(double[][] phi)
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
					betaTtheta += phi[k][doc.ids[n]]*theta[m][k];
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
			String path_res = new File(path, "res_" + num_topics).getAbsolutePath();
			File eval = new File(path_res, "gibbs_eval"); 
			File all_eval = new File(path, "gibbs_eval");
			String s = path_res + " : " + perplex + "\n";
			FileUtils.writeStringToFile(eval, sb.toString());
			FileUtils.writeStringToFile(all_eval, s, true);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void computePerplexity(double[][] phi, int[][] topwords)
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
				if(flag)
				{
					double betaTtheta = 0;
					for(int k = 0; k < num_topics; k++)
					{
						betaTtheta += phi[k][doc.ids[n]]*theta[m][k];
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
			String path_res = new File(path, "res_" + num_topics).getAbsolutePath();
			File eval = new File(path_res, "gibbs_eval"); 
			File all_eval = new File(path, "gibbs_eval");
			String s = path_res + " : " + perplex + "\n";
			FileUtils.writeStringToFile(eval, sb.toString());
			FileUtils.writeStringToFile(all_eval, s, true);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
