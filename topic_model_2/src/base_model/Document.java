package base_model;


import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.apache.commons.io.FileUtils;

public class Document {	
	public String path;  //the running path
	public String[] words;
    public int[] counts;
    public int[] ids;
    public int[] z;     //Topic assignment
    public int length;  //Total unique words
    public int total;   //Total words
    public String doc_name;
    public Map<Integer, Integer> wordCount = null;
    public Map<Integer, Integer> idToIndex = null;
    public Map<Integer, List<Integer>> adj = null;   //adjacent list of nodes
    public Map<Integer, List<Integer>> adj2 = null;  //distance two adjacent list of nodes
    public double zeta1; //Taylor approx
    public double zeta2;
    public double exp_ec;  //expectation of coherent edges
    public double exp_ec2;
    public int num_e;   //total number of edges
    public int num_e2;  //total number of edges with distance 2
    public double exp_theta_square;
    
    public double[] gamma;  //variational dirichlet parameter, K dimension  initialized when run EM
    public double[][] phi; //variational multinomial, corpus.maxLength() * K dimension
    
    public Vocabulary voc;
    
    public Document(String path, String doc_name, Vocabulary voc)
    {
    	this.path = path;
    	this.doc_name = doc_name;
    	this.voc = voc;
    	wordCount = new TreeMap<Integer, Integer>();
    	idToIndex = new TreeMap<Integer, Integer>();
    	adj = new TreeMap<Integer, List<Integer>>();
    	adj2 = new TreeMap<Integer, List<Integer>>();
    }
        
    
    /**
     * format to  word: count and initialize each doc object, set word count map, set words, ids, counts array
     * @param voc
     */
    public void formatDocument() 
    {
    	String text = "";
    	try {
    		String data_words = new File(path, "data_words").getAbsolutePath();
			text = FileUtils.readFileToString(new File(data_words, doc_name));
		} catch (IOException e) {
			e.printStackTrace();
		}
    	String[] ws = text.split(" ");
    	for(String word: ws)  //put word count pair to map
    	{
    		if(!voc.wordToId.containsKey(word))
    			continue;
    		int id = voc.wordToId.get(word);
    		if(!wordCount.containsKey(id))
    		{
    			wordCount.put(id, 1);
    		}
    		else
    		{
    			wordCount.put(id, wordCount.get(id) + 1);
    		}
    	}
    	this.length = wordCount.size();
    	//Initialize word topic assignment to -1. topic assignment ranges from 0 to k-1 
    	this.z = new int[this.length];
    	for(int i = 0; i < length; i++)
    		z[i] = -1;
    	words = new String[wordCount.size()];
    	counts = new int[wordCount.size()];
    	ids = new int[wordCount.size()];
    	int i = 0;
    	for (Map.Entry<Integer, Integer> entry : wordCount.entrySet())
		{    		
			int id = entry.getKey();
			int count = entry.getValue();
			words[i] = voc.idToWord.get(id);
			counts[i] = count;
			ids[i] = id;
			idToIndex.put(id, i);
			i++;
			this.total += count;
		}
    }
    
   /**
    * get adjacent words for each word from data_edges folder
    */
    public void getEdges2()
    {
    	String text = "";
    	String path_edges = new File(path, "data_edges").getAbsolutePath();
		try {
			text = FileUtils.readFileToString(new File(path_edges, doc_name));
		} catch (IOException e) {
			e.printStackTrace();
		}
		if(text.equals(""))  //No adj nodes
			return;
		for(String line: text.split("\\r?\\n"))
		{
			String word = line.substring(0, line.indexOf(':'));
			List<String> list = Arrays.asList(line.substring(line.indexOf('[') + 1, line.indexOf(']')).split(","));
			num_e += list.size();
			if(!voc.wordToId.containsKey(word))
    			continue;
			int wordid = voc.wordToId.get(word);
			List<Integer> adjList = new ArrayList<Integer>();
			for(String w: list)
			{
				if(!voc.wordToId.containsKey(w.trim()))
					continue;
				adjList.add(voc.wordToId.get(w.trim()));
			}
			adj.put(wordid, adjList);
		}
		num_e = num_e/2; //we count num_e twice;
    }
    
    public void getEdges3()
    {
    	for (Map.Entry<Integer, List<Integer>> entry : adj.entrySet())
    	{
    		List<Integer> adjlist2 = new ArrayList<Integer>(); //store distance 2 nodes
    		List<Integer> adjlist1 = entry.getValue();  //adjacent nodes
    		int wordid = entry.getKey();
    		for(int adjid: adjlist1)
    		{
    			for(int adj2id:adj.get(adjid))
    			{
    				if(adj2id != wordid && !adjlist2.contains(adj2id) && voc.idToWord.containsKey(adj2id))
    				{
    					adjlist2.add(adj2id);
    					num_e2++;
    				}
    			}
    		}
    		adj2.put(wordid, adjlist2);
    	}
    	num_e2 = num_e2/2;
    	
    }


}
