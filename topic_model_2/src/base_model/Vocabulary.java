package base_model;


import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.apache.commons.io.FileUtils;

public class Vocabulary {
	public Map<Integer, String> idToWord = null;
	public Map<String, Integer> wordToId = null;
	public Map<String, Integer> wordCount = null;
	
	
	public Vocabulary()
	{
		idToWord = new TreeMap<Integer, String>();
		wordToId = new TreeMap<String, Integer>();
		wordCount = new TreeMap<String, Integer>();
		
	}
	
	public int size()
	{
		return wordCount.size();
	}
	
	/**
	 * This method return the vocabulary of the whole corpus
	 * @param path         the bag of words document path, in data_words folder 
	 * @param min_count    remove any word which counts less than min_count
	 */
	public void getVocabulary(String path, int min_count, int max_count)
	{
		File voc_file = new File(path, "idAndWord" + "_" + min_count + "_" + max_count);
		File count_file = new File(path, "idAndWord" + "_" + min_count + "_" + max_count + "_count");
		if(voc_file.exists())  //If vocabulary file already exists, read voc from file
		{
			String text = "";
    		try {
    			text = FileUtils.readFileToString(voc_file);
    		} catch (IOException e) {
    			e.printStackTrace();
    		}
    		String[] lines = text.split("\\r?\\n");
    		for(String line : lines)
    		{
        		int id = Integer.parseInt(line.substring(0, line.indexOf(':')));
        		String word = line.substring(line.indexOf(":") + 1);
        		idToWord.put(id, word);
    			wordToId.put(word, id);
    		}
    		    		
    		try {
    			text = FileUtils.readFileToString(count_file);
    		} catch (IOException e) {
    			e.printStackTrace();
    		}
    		String[] ls = text.split("\\r?\\n");
    		for(String line: ls)
    		{
        		int count = Integer.parseInt(line.substring(line.indexOf(":") + 1));
        		String word = line.substring(0, line.indexOf(':'));
        		wordCount.put(word, count);
    		}    		
		}
		else  //else create vocabulary, id word and word counts
		{
			List<File> dir = process.Preprocess.listDir(new File(path, "data_words").getAbsolutePath());
			
			//Calculate words counts
	    	for(File d : dir)
	    	{
	    		String text = "";
	    		try {
	    			text = FileUtils.readFileToString(d);
	    		} catch (IOException e) {
	    			e.printStackTrace();
	    		}
	
	    		String[] words = text.split(" ");
	    		for(String word: words)
	    		{
	    			if(wordCount.containsKey(word))
	    			{
	    				wordCount.put(word, wordCount.get(word) + 1);
	    			}
	    			else
	    			{
	    				wordCount.put(word, 1);	
	    			}
	    		}
	    	}
	    	    	
	    	//Remove word counts pair which counts less than min_count, then create id word map
	    	//Need to create a copy of map, otherwise concurrent exception    	    	
	    	Map<String, Integer> temp_wordCount = new TreeMap<String, Integer>(wordCount);
	    	int id = 0;
	    	for (Map.Entry<String, Integer> entry : temp_wordCount.entrySet())
			{
	    		int count = entry.getValue();
	    		String word = entry.getKey();
	    		if(count < min_count || count > max_count)
	    		{
	    			wordCount.remove(word);    			
	    		}
	    		else
	    		{
	    			idToWord.put(id, word);
	    			wordToId.put(word, id);
	    			id++;
	    		}
			}
	    	printToFile(new File(path, "idAndWord" + "_" + min_count + "_" + max_count).getAbsolutePath());
		}
	}

	
	public void printToFile(String filepath) {
		StringBuilder sb = new StringBuilder();
		for (Map.Entry<Integer, String> entry : idToWord.entrySet())
		{
			int id = entry.getKey();
			String word = entry.getValue();
			sb.append(id + ":" + word);
			sb.append("\n");
		}
		try {
			FileUtils.writeStringToFile(new File(filepath), sb.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}
		sb = new StringBuilder();
		for (Map.Entry<String, Integer> entry : wordCount.entrySet())
		{
			int count = entry.getValue();
			String word = entry.getKey();
			sb.append(word + ":" + count);
			sb.append("\n");
		}
		try {
			FileUtils.writeStringToFile(new File(filepath + "_count"), sb.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
}

