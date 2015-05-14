package base_model;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.commons.io.FileUtils;

import process.Preprocess;


public class Corpus {
	public Document[] docs;          
	public Document[] docs_test;
	public int num_terms;            //V
    public int num_docs;             //M
    public int num_docs_test;
    public Vocabulary voc;
    public String path;
    
    //The files in this path are already tokenized and removed stop words in Python
    /**
     * create corpus. Before doing this, we already preprocess documents into data_words, data_trees, data_edges folder.
     * @param path             running folder, the parent folder of everything
     * @param min_count        any word which counts less than min_count will be removed
     * @param percentage       the percentage of training data
     * @param type             "LDA", "GTRF", "MGTRF"
     */
    public Corpus(String path, int min_count, int max_count, double percentage, String type)
    {
    	StringBuilder sb = new StringBuilder();
    	this.path = path;
    	// Iterate all files and get vocabulary, word id maps.
    	voc = new Vocabulary();
    	voc.getVocabulary(path, min_count, max_count);
    	num_terms = voc.size();
    	System.out.println("number of terms   :" + num_terms);
    	sb.append("number of terms   :" + num_terms);
    	sb.append("\n");
    	
    	List<File> files = Preprocess.listDir(new File(path, "data_words").getAbsolutePath());
    	int num = files.size();
    	System.out.println("number of docs   :" + num);
    	sb.append("number of docs   :" + num);
    	sb.append("\n");
    	num_docs = (int) Math.round(files.size()*percentage);
    	System.out.println("number of training docs    :" + num_docs);
    	sb.append("number of training docs    :" + num_docs);
    	sb.append("\n");
    	
    	docs = new Document[num_docs];
    	int i = 0;
    	System.out.println("=======process training set========");
		while(i < num_docs)
		{
			if(i%10 == 0)
				System.out.println("Loading document " + i);
			Document doc = new Document(path, files.get(i).getName(), voc);
			doc.formatDocument(); //format document to word: count, and set words, counts, ids array
//			System.out.println("Document " + d + " contain unique words : " + doc.length);
			if(type.equals("GTRF"))
			{
				doc.getEdges2();
			}
			if(type.equals("MGTRF"))
			{
				doc.getEdges2();
				doc.getEdges3();
			}
			docs[i] = doc;
			i++;
		}
		
		num_docs_test = num - num_docs;
		System.out.println("number of test docs    :" + num_docs_test);
		sb.append("number of testing docs    :" + num_docs_test);
    	sb.append("\n");
    	docs_test = new Document[num_docs_test];
		System.out.println("=======process test set========");
		int j = 0;
		while(j < num_docs_test)
		{
			if(j%10 == 0)
				System.out.println("Loading document " + j);
			Document doc = new Document(path, files.get(i).getName(), voc);
			doc.formatDocument(); //format document to word: count, and set words, counts, ids array
//			System.out.println("Document " + d + " contain unique words : " + doc.length);
			if(type.equals("GTRF"))
			{
				doc.getEdges2();
			}
			if(type.equals("MGTRF"))
			{
				doc.getEdges2();
				doc.getEdges3();
			}
			docs_test[j] = doc;
			j++;
			i++;
		}
		
		try {
			FileUtils.writeStringToFile(new File(path, "corpus_info"), sb.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}
    }
    
    public int maxLength()
    {
    	int max = 0;
    	for(int i = 0; i < docs.length; i++)
    	{
//    		System.out.println(docs[i].doc_name);
    		max = max > docs[i].length?max:docs[i].length;
    	}
    	return max;
    }

}
