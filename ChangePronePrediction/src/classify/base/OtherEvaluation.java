package classify.base;

import java.util.Random;

import preprocess.Sample;


import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Range;

public class OtherEvaluation extends Evaluation {
	int choose;

	public OtherEvaluation(Instances data, int choose) throws Exception {
		super(data);
		this.choose = choose;
	}

	public void crossValidateModel(Classifier classifier, Instances data,
			int numFolds, Random random, Object... forPredictionsPrinting)
			throws Exception {

		// Make a copy of the data we can reorder
		data = new Instances(data);
		data.randomize(random);
		if (data.classAttribute().isNominal()) {
			data.stratify(numFolds);
		}

		// We assume that the first element is a StringBuffer, the second a
		// Range
		// (attributes
		// to output) and the third a Boolean (whether or not to output a
		// distribution instead
		// of just a classification)
		if (forPredictionsPrinting.length > 0) {
			// print the header first
			StringBuffer buff = (StringBuffer) forPredictionsPrinting[0];
			Range attsToOutput = (Range) forPredictionsPrinting[1];
			boolean printDist = ((Boolean) forPredictionsPrinting[2])
					.booleanValue();
			printClassificationsHeader(data, attsToOutput, printDist, buff);
		}
		Sample sample = new Sample();
		// Do the folds
		for (int i = 0; i < numFolds; i++) {
			Instances trainOrigin = data.trainCV(numFolds, i, random);
			Instances train=null;
			switch(choose){
				case 0:{
					train=trainOrigin;
					break;
				}
				case 1:{
					train=sample.UnderSample(trainOrigin);
					break;
				}
				case 2:{
					train=sample.OverSample(trainOrigin);
					break;
				}
				case 3:{
					train=sample.AntiUnderSample(trainOrigin,0.1);
					break;
				}
				case 4:{
					train=sample.AntiOverSample(trainOrigin,0.1);
					break;
				}
				case 5:{
					train=sample.SmoteSample(trainOrigin, 1);
					break;
				}
			}
			setPriors(train);
			Classifier copiedClassifier = Classifier.makeCopy(classifier);
			copiedClassifier.buildClassifier(train);
			Instances test = data.testCV(numFolds, i);
			evaluateModel(copiedClassifier, test, forPredictionsPrinting);
		}
		m_NumFolds = numFolds;
	}

}
