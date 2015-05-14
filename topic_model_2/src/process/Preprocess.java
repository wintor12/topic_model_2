package process;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.TreeMap;

import org.apache.commons.io.FileUtils;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.Sentence;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.trees.GrammaticalStructureFactory;
import edu.stanford.nlp.trees.PennTreebankLanguagePack;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreebankLanguagePack;
import edu.stanford.nlp.trees.TypedDependency;
import edu.stanford.nlp.util.CoreMap;

public class Preprocess {
	
	/**
	 * Before we run any LDA programs, we need to pre-process documents first.
	 * The constructor only takes run path(running directory), data path(original documents in a single folder)
	 * and stop words path. Then it will automatically create data_words(bag of words document), data_tree(dependency tree)
	 * and data_edge(edges for each node) during the process of 2, 3.
	 * The order to run preprocess is: 
	 * 1 constructor
	 * 2 getWords/getwordsandTrees
	 * 3 findEdges
	 */
	
	String path_documents; //The original data set folder
	String path;  //The original data set folder
	String path_words;  //the bag of words document, each word split by " "
	String path_trees;  //the dependency tree of a document, each wordNode split by \t
	String path_edges;  //the edges of a tree document, has format term:[term1, term2]
	String path_train;
	String path_test;
	String path_sentences;
	List<String> stopwords;
	
	public Preprocess(String run_path, String data_path, String stopwords_path)
	{
		this.path_documents = data_path;
		this.path = run_path;
		this.path_words = new File(run_path, "data_words").getAbsolutePath();
		this.path_trees = new File(run_path, "data_trees").getAbsolutePath();
		this.path_edges = new File(run_path, "data_edges").getAbsolutePath();
		this.path_sentences = new File(run_path, "data_sentences").getAbsolutePath();
		this.path_train = new File(run_path, "training").getAbsolutePath();
		this.path_test = new File(run_path, "testing").getAbsolutePath();
		String text = "";
		try {
			text = FileUtils.readFileToString(new File(stopwords_path));
		} catch (IOException e) {
			e.printStackTrace();
		}
		String[] words = text.split(" ");
		stopwords = Arrays.asList(words);
	}
	
	/**
	 * Generate bag of words documents from "path_documents" 
	 * Output to path_words
	 */
	public void getSentences() {
		Properties props = new Properties();
		props.put("annotators", "tokenize, ssplit");
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		for (File file : listDir(path_documents)) {
			String name = file.getName();
			System.out.println(name);
			StringBuilder sb = new StringBuilder(); // bag of words
			String text = "";
			try {
				text = FileUtils.readFileToString(file);
			} catch (IOException e) {
				e.printStackTrace();
			}
			Annotation document = new Annotation(text);
			pipeline.annotate(document);
			List<CoreMap> sentences = document.get(SentencesAnnotation.class);
			for (CoreMap s : sentences) {
				for (CoreLabel token : s.get(TokensAnnotation.class)) {
					String word = token.toString().toLowerCase();
					if (word.matches("[a-z]+")) {
						if (!stopwords.contains(word))
							sb.append(word + " ");
					}
				}
				sb.append("\n");
			}
			try {
				FileUtils.writeStringToFile(new File(path_sentences, name),
						sb.toString());
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
	/**
	 * Generate bag of words documents from "path_documents" 
	 * Output to path_words
	 */
	public void getWords() {
		Properties props = new Properties();
		props.put("annotators", "tokenize, ssplit");
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		for (File file : listDir(path_documents)) {
			String name = file.getName();
			System.out.println(name);
			StringBuilder sb = new StringBuilder(); // bag of words
			String text = "";
			try {
				text = FileUtils.readFileToString(file);
			} catch (IOException e) {
				e.printStackTrace();
			}
			Annotation document = new Annotation(text);
			pipeline.annotate(document);
			List<CoreMap> sentences = document.get(SentencesAnnotation.class);
			for (CoreMap s : sentences) {
				for (CoreLabel token : s.get(TokensAnnotation.class)) {
					String word = token.toString().toLowerCase();
					if (word.matches("[a-z]+")) {
						if (!stopwords.contains(word))
							sb.append(word + " ");
					}
				}
			}
			try {
				FileUtils.writeStringToFile(new File(path_words, name),
						sb.toString());
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
	/**
	 * Generate bag of words documents and dependency trees documents from "path_documents" 
	 * Output to path_words and path_trees
	 */
	public void getWordsAndTrees() {
		LexicalizedParser lp = LexicalizedParser.loadModel(
				"edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz",
				"-maxLength", "80", "-retainTmpSubcategories");
		TreebankLanguagePack tlp = new PennTreebankLanguagePack();
		GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
		Properties props = new Properties();
		props.put("annotators", "tokenize, ssplit");
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		for (File file : listDir(path_documents)) {			
			String name = file.getName();
			System.out.println(name);
			File t = new File(path_trees, name);
			if(t.exists())
			{
				System.out.println("exist!");
				continue;
			}
			StringBuilder sb = new StringBuilder(); // bag of words
			StringBuilder sb2 = new StringBuilder(); // trees
			String text = "";
			try {
				text = FileUtils.readFileToString(file);
			} catch (IOException e) {
				e.printStackTrace();
			}
			Annotation document = new Annotation(text);
			pipeline.annotate(document);
			List<CoreMap> sentences = document.get(SentencesAnnotation.class);
			for (CoreMap s : sentences) {
				ArrayList<String> sent = new ArrayList<String>();
				for (CoreLabel token : s.get(TokensAnnotation.class)) {
					String word = token.toString().toLowerCase();
					if (word.matches("[a-z]+")) {
						if (!stopwords.contains(word))
							sb.append(word + " ");
						sent.add(token.get(TextAnnotation.class));
					}
				}
				try {
					String[] sentence = sent.toArray(new String[sent.size()]);
					Tree parse = lp.apply(Sentence.toWordList(sentence));
					GrammaticalStructure gs = gsf
							.newGrammaticalStructure(parse);
					Collection<TypedDependency> tdl = gs
							.typedDependenciesCCprocessed();
					// System.out.println(tdl);
					Iterator it = tdl.iterator();
					while (it.hasNext()) {
						String wordnode = it.next().toString();
						sb2.append(wordnode.substring(wordnode.indexOf('('),
								wordnode.indexOf(')') + 1) + "\t");
					}
					sb2.append("\n");
				} catch (Exception e) {
					System.out.println("error sentence:" + s.toString());
				}
			}
			try {
				FileUtils.writeStringToFile(new File(path_words, name),
						sb.toString());
				FileUtils.writeStringToFile(new File(path_trees, name),
						sb2.toString());
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
	/**
	 * @param wordNode  has the format eg, (CALL-5, Robert-2) from dependency tree parsing
	 * @return  parent word, This example is "Call"
	 */
	public String extractParent(String wordNode)
	{
		return wordNode.substring(1, wordNode.indexOf('-'));
	}
	
	/**
	 * return the word "Robert" of a wordNode  eg, (CALL-5, Robert-2) from dependency tree parsing
	 * @param wordNode
	 * @return 
	 */
	public String extractWord(String wordNode)
	{
		return wordNode.substring(wordNode.indexOf(',') + 2, wordNode.indexOf('-', wordNode.indexOf(',')));
	}
	
	/**
	 * Find edges of each tree document in path_trees folder, and output in path_edge folder
	 */
	public void findEdges()
	{
		for(File file: listDir(path_trees))
		{
			Map<String, List<String>> map = new TreeMap<String, List<String>>();
			findEdgesFromTreeDoc(file, map);
		}
		
	}
	
	/** 
	 * this method can process one tree document in data_tree and output tree edges document in data_edges
	 * @param file   has to be dependency tree file in data_tree
	 * @param map    a map that key is a word in the corpus, value is a List contains all words has distance 1 with this word
	 */
	public void findEdgesFromTreeDoc(File file, Map<String, List<String>> map)
	{
		String name = file.getName();
		String worddict = "";
		try {
			worddict = FileUtils.readFileToString(new File(path_words, name));
		} catch (IOException e) {
			e.printStackTrace();
		}
		Set<String> dict = new HashSet<String>();   //word dictionary of current document
		for(String w: worddict.split(" "))
		{
			dict.add(w);
		}
		String text = "";
		try {
			text = FileUtils.readFileToString(new File(path_trees, name));
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		StringBuilder sb = new StringBuilder();
		String[] sentences = text.split("\\r?\\n");
		for(String sent: sentences)
		{
			String[] wordNodes = sent.split("\t");
			for(String wordNode: wordNodes)
			{
				String word = extractWord(wordNode).toLowerCase();
				String parent = extractParent(wordNode).toLowerCase();
				//If words are not in dict, remove it
				if(parent.equals("ROOT") || !dict.contains(parent) || !dict.contains(word))
					continue;
				if(!map.containsKey(word))
					map.put(word, new ArrayList<String>());
				if(!map.containsKey(parent))
					map.put(parent, new ArrayList<String>());
				if(!map.get(parent).contains(word))
					map.get(parent).add(word);
				if(!map.get(word).contains(parent))
					map.get(word).add(parent);			
			}
		}
		for (Map.Entry<String, List<String>> entry : map.entrySet())
		{
			String term = entry.getKey();
			List<String> adj_term = entry.getValue();
			sb.append(term + ":" + adj_term.toString());
			sb.append("\n");
		}
		try {
			FileUtils.writeStringToFile(new File(path_edges, name), sb.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Do not use any more
	 * @param percentage
	 */
	public void selectTrainingAndTestingData(double percentage)
	{
		File train = new File(this.path_train);
		File test = new File(this.path_test);
		List<File> files = listDir(this.path_documents);
		int num_train = (int) Math.round(files.size()*percentage);
		StringBuilder sb = new StringBuilder();
		StringBuilder sb2 = new StringBuilder();
		int i = 0;
		for(i = 0; i < num_train; i++)
		{			
			sb.append(files.get(i).getName());
			sb.append("\n");
		}
		for(i = num_train; i < files.size(); i++)
		{			
			sb2.append(files.get(i).getName());
			sb2.append("\n");
		}
		try {
			FileUtils.writeStringToFile(train, sb.toString());
			FileUtils.writeStringToFile(test, sb2.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/*
	 * list files in a folder
	 */	
	public static List<File> listDir(String folder)
	{
		File f = new File(folder);
		File fe[] = f.listFiles();
		if(fe == null)
		{
			System.out.println("No such directory!!");
			return null;
		}
		List<File> files = new LinkedList<File>(Arrays.asList(fe));
		for(int i = 0; i < files.size(); i++)
		{
			File file = files.get(i);
			if (!file.isFile() || file.isHidden() || file.getName().startsWith(".")) 
			{
				files.remove(i);
			}
		}
		return files;
	}

}
