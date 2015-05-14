package process;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;

import sgtrf.EM_s;
import sgtrf.EM_s2;
import sgtrf.EM_s3;
import sgtrf.EM_s4;
import mgtrf.EM_m;
import gtrf.EM_g;
import base_model.Corpus;
import base_model.EM;
import base_model.GibbsSampling;

public class Test {
	
	//The LDA output running folder. All folders and files created here during running topic model 
	public static String run_path = "C:\\Exp\\lda\\nips1\\";
//	public static String run_path = "C:\\Exp\\lda\\nips1\\";
	//The original data set folder
//	public static String data_path = "C:\\Exp\\lda\\20news\\data\\";
//	public static String data_path = "C:\\Exp\\lda\\nips1\\data\\";
    //The stop words folder
	public static String stopwords_path = "C:\\Exp\\lda\\stopwords2.txt";

//	public static String run_path = "/Users/tongwang/Desktop/exp/lda/nips2/";  
//	public static String data_path = "/Users/tongwang/Desktop/exp/lda/nips2/";
//	public static String stopwords_path = "/Users/tongwang/Desktop/exp/lda/stopwords2.txt";
	
	public static void main(String[] args) {
		
//		Preprocess p = new Preprocess(run_path, data_path, stopwords_path);
//		p.getSentences();
//		p.getWords();
//		p.getWordsAndTrees();
//		p.findEdges();
		
		double alpha = 0.1;
		double beta = 0.1;
		int min_count = 20;
		int max_count = 1000;
		double train_percentage = 0.8;
		double lambda2 = 0.2;
		double lambda4 = 0.5;
		int K = 2;
		int[] ks = {10,15,20,25};
		int iters = 3;
		double[] ls = {0.6,0.8, 1.2};
		//folder path is the running path, path_res is the output result path

		for(int i = 0; i < ks.length; i++){
			for(int iter = 0; iter < iters; iter++)
			{
				Corpus corpus = new Corpus(run_path, min_count, max_count, train_percentage, "LDA");
				String path_res = new File(run_path, "res_" + ks[i]).getAbsolutePath();	
				EM em = new EM(run_path, path_res, ks[i], corpus, beta);
				em.run_em("LDA");
			}
		}
		
//		for(int i = 2; i < 21; i++){
//		Corpus corpus = new Corpus(run_path, min_count, train_percentage, "LDA");
//		String path_res = new File(run_path, "res_" + i).getAbsolutePath();
//		GibbsSampling g = new GibbsSampling(run_path, path_res, i, corpus, 0, alpha, beta, 10, 500, 1000);
//		g.run_gibbs();
//		}
		
				
//		//GTRF		
//		for(int i = 0; i < ks.length; i++){
//			for(int iter = 0; iter < iters; iter++)
//			{
//				Corpus corpus2 = new Corpus(run_path, min_count, max_count, train_percentage, "GTRF");
//				String path_res2 = new File(run_path, "res_" + ks[i] + "_" + lambda2).getAbsolutePath();
//				EM_g em2 = new EM_g(run_path, path_res2, ks[i], corpus2, beta, lambda2);
//				em2.run_em("GTRF");	
//				
//			}
//		}
//		//EGTRF
//		for(int i = 0; i < ks.length; i++){
//			for(int j = 0; j < ls.length; j++){
//				for(int iter = 0; iter < iters; iter++)
//				{
//					lambda4 = ls[j];
//					Corpus corpus3 = new Corpus(run_path, min_count, max_count, train_percentage, "MGTRF");
//					String path_res3 = new File(run_path, "res_" + ks[i] + "_" + lambda2 + "_" + lambda4).getAbsolutePath();
//					EM_m em3 = new EM_m(run_path, path_res3, ks[i], corpus3, beta, lambda2, lambda4);
//					em3.run_em("MGTRF");
//				}
//			}
//		}
		
//		Corpus corpus = new Corpus(run_path, min_count, max_count, train_percentage, "MGTRF");
//		double[][] sim = init_sim(new File(run_path, "sim_matrix").getAbsolutePath(), corpus);
//		for(int i = 0; i < ks.length; i++){	
//			for(int iter = 0; iter < iters; iter++)
//			{
//				Corpus corpus5 = new Corpus(run_path, min_count, max_count, train_percentage, "GTRF");
//				String path_res5 = new File(run_path, "res_" + ks[i] + "_" + lambda2).getAbsolutePath();
//				EM_s2 em2 = new EM_s2(run_path, path_res5, ks[i], corpus5, beta, lambda2, sim);
//				em2.run_em("GTRF");	
//			}
//		}
//		for(int i = 0; i < ks.length; i++){			
//			for(int j = 0; j < ls.length; j++){
//				for(int iter = 0; iter < iters; iter++)
//				{
//					lambda4 = ls[j];
//					Corpus corpus4 = new Corpus(run_path, min_count, max_count, train_percentage, "MGTRF");
//					String path_res4 = new File(run_path, "res_" + ks[i] + "_" + lambda2 + "_" + lambda4).getAbsolutePath();
//					EM_s em4 = new EM_s(run_path, path_res4, ks[i], corpus4, beta, lambda2, lambda4, sim);
//					em4.run_em("MGTRF");
//				}
//			}
//		}
//		
//		Corpus corpus = new Corpus(run_path, min_count, max_count, train_percentage, "MGTRF");
//		double[][] sim = init_sim(new File(run_path, "sim_matrix").getAbsolutePath(), corpus);
//		for(int i = 0; i < ks.length; i++){			
//			Corpus corpus5 = new Corpus(run_path, min_count, max_count, train_percentage, "GTRF");
//			String path_res5 = new File(run_path, "res_" + ks[i] + "_" + lambda2).getAbsolutePath();
//			EM_s4 em2 = new EM_s4(run_path, path_res5, ks[i], corpus5, beta, lambda2, sim);
//			em2.run_em("GTRF");
//			
//			for(int j = 0; j < ls.length; j++){
//				lambda4 = ls[j];
//				Corpus corpus4 = new Corpus(run_path, min_count, max_count, train_percentage, "MGTRF");
//				String path_res4 = new File(run_path, "res_" + ks[i] + "_" + lambda2 + "_" + lambda4).getAbsolutePath();
//				EM_s3 em4 = new EM_s3(run_path, path_res4, ks[i], corpus4, beta, lambda2, lambda4, sim);
//				em4.run_em("MGTRF");
//			}
//		}
	}
	
	public static double[][] init_sim(String path, Corpus corpus)
	{
		int size = corpus.voc.size();
		double[][] sim = new double[size][size];
		for(int i = 0; i < size; i++)
		{
			if(i % 100 == 0)
				System.out.println("word: " + i);
			String text = "";
			try {
				text = FileUtils.readFileToString(new File(path, i + ""));
			} catch (IOException e) {
				e.printStackTrace();
			}
			String[] values = text.split(" ");
			for(int j = 0; j < size; j++)
			{
				sim[i][j] = Double.parseDouble(values[j]);
			}
		}
		return sim;
	}
	

}
