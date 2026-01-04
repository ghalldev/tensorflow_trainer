package gh.proto.tensorflow.work;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Component;
import org.tensorflow.Graph;
import org.tensorflow.Result;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.SessionFunction;
import org.tensorflow.Signature;
import org.tensorflow.Tensor;
import org.tensorflow.framework.initializers.Glorot;
import org.tensorflow.framework.initializers.VarianceScaling.Distribution;
import org.tensorflow.framework.losses.MeanSquaredError;
import org.tensorflow.framework.losses.Reduction;
import org.tensorflow.framework.optimizers.Adam;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.types.TFloat32;

import gh.proto.tensorflow.ProtoWorkException;
import jakarta.annotation.PostConstruct;

@Component
public class IrisTrainer {

    private final Logger logger = LogManager.getLogger(getClass());

    private static final String CSV_SEPARATOR_REGEX = ",";

    /**
     * number of records in the input csv file with the data
     */
    private static final int INPUT_DATA_LINES = 150;

    /**
     * set by number of inputs: 1. sepal length, 2. sepal width, 3. petal length ,
     * 4. petal width
     */
    private static final long INPUT_LAYER_WIDTH = 4L;

    private static final long HIDDEN_LAYER_1_WIDTH = 5L;
    private static final long HIDDEN_LAYER_2_WIDTH = 4L;

    // discovered empirically
    private static final float LEARNING_RATE = 0.01F;
    private static final int TRAINING_EPOCHS = 4;
    private static final long RANDOM_SEED = 1234567L;

    /**
     * set by number of features: 1. Iris-setosa, 2. Iris-versicolor, 3.
     * Iris-virginica
     */
    private static final long OUTPUT_LAYER_WIDTH = 3L;

    // must match the order from the input csv file with the
    // data(train_data/iris/bezdekIris.data)
    private static final int CSV_SEPAL_LENGTH_IDX = 0;
    private static final int CSV_SEPAL_WIDTH_IDX = 1;
    private static final int CSV_PETAL_LENGTH_IDX = 2;
    private static final int CSV_PETAL_WIDTH_IDX = 3;
    private static final int CSV_SPECIES_NAME_IDX = 4;

    // chosen by us and must be consistent thru training and running
    private static final int OUTPUT_IRIS_SETOSA_IDX = 0;
    private static final int OUTPUT_IRIS_VERSICOLOUR_IDX = 1;
    private static final int OUTPUT_IRIS_VIRGINICA_IDX = 2;

    private static final String OP_NAME_INPUT_LAYER_PLACEHOLDER = "inputLayerPlaceholder";
    private static final String OP_NAME_HIDDEN_LAYER1_WEIGHTS = "hiddenLayer1Weights";
    private static final String OP_NAME_HIDDEN_LAYER1_BIASES = "hiddenLayer1Biases";
    private static final String OP_NAME_HIDDEN_LAYER2_WEIGHTS = "hiddenLayer2Weights";
    private static final String OP_NAME_HIDDEN_LAYER2_BIASES = "hiddenLayer2Biases";
    private static final String OP_NAME_OUTPUT_LAYER_BIASES = "outputLayerBiases";
    private static final String OP_NAME_OUTPUT_LAYER_WEIGHTS = "outputLayerWeights";
    private static final String OP_NAME_OUTPUT_ACTIVATION = "outputActivation";

    // Iris data downloaded from: https://archive.ics.uci.edu/dataset/53/iris
    @Value("classpath:train_data/iris/bezdekIris.data")
    private Resource inputData;

    @Value("${tensorflow.iris.path}")
    private String exportPath;

    private enum IrisSpecies {
        IRIS_SETOSA, IRIS_VERSICOLOUR, IRIS_VIRGINICA;

        static IrisSpecies getIrisSpecies(String speciesName) {
            switch (speciesName) {
            case "Iris-setosa":
                return IRIS_SETOSA;
            case "Iris-versicolor":
                return IRIS_VERSICOLOUR;
            case "Iris-virginica":
                return IRIS_VIRGINICA;
            default:
                throw new ProtoWorkException("Unknown species name: " + speciesName);
            }
        }
    };

    private record IrisDataLine(float sepalLength, float sepalWidth, float petalLength, float petalWidth,
            IrisSpecies irisSpecies) {
    };

    @PostConstruct
    public void doTrain() {
        try (Graph graph = new Graph(); Session session = new Session(graph)) {
            train(graph, session);
            save(session);
        } catch (IOException e) {
            throw new ProtoWorkException("Can't train/save", e);
        }
    }

    public void save(Session tfSession) throws IOException {
        Signature signature = Signature.builder().key(Signature.DEFAULT_KEY)
                .input(OP_NAME_INPUT_LAYER_PLACEHOLDER,
                        tfSession.graph().operation(OP_NAME_INPUT_LAYER_PLACEHOLDER).output(0))
                .output(OP_NAME_OUTPUT_ACTIVATION, tfSession.graph().operation(OP_NAME_OUTPUT_ACTIVATION).output(0))
                .build();
        SessionFunction sessionFunction = SessionFunction.create(signature, tfSession);
        SavedModelBundle.exporter(exportPath).withFunction(sessionFunction).withTags(SavedModelBundle.DEFAULT_TAG)
                .export();
    }

    public void train(Graph tfGraph, Session tfSession) throws IOException {
        var trainData = readTrainData();
        // very important to shuffle because the grouping of the data in the input csv
        // can appear as an unwanted pattern during training
        Collections.shuffle(trainData, new Random(RANDOM_SEED));

        var tensorFlowApi = Ops.create(tfGraph);
        buildNetwork(tensorFlowApi);

        // loss and optimizer only needed during training
        var meanSquaredErrorLoss = new MeanSquaredError(Reduction.AUTO);
        var optimizer = new Adam(tfGraph, LEARNING_RATE);

        var trainingOutputPlaceholder = tensorFlowApi.placeholder(TFloat32.class,
                Placeholder.shape(Shape.of(-1, OUTPUT_LAYER_WIDTH)));
        var minimize = optimizer.minimize(meanSquaredErrorLoss.call(tensorFlowApi, trainingOutputPlaceholder,
                tfGraph.operation(OP_NAME_OUTPUT_ACTIVATION).output(0)));

        for (int currentTrainingEpoch = 0; currentTrainingEpoch < TRAINING_EPOCHS; currentTrainingEpoch++) {
            var numberOfPredictedOk = 0;
            for (int inputDataIdx = 0; inputDataIdx < trainData.size(); inputDataIdx++) {
                var currentInputData = trainData.get(inputDataIdx);
                try (var inputDataTensor = Tensor.of(TFloat32.class, Shape.of(1, INPUT_LAYER_WIDTH), data -> {
                    data.setFloat(currentInputData.sepalLength, 0, CSV_SEPAL_LENGTH_IDX);
                    data.setFloat(currentInputData.sepalWidth, 0, CSV_SEPAL_WIDTH_IDX);
                    data.setFloat(currentInputData.petalLength, 0, CSV_PETAL_LENGTH_IDX);
                    data.setFloat(currentInputData.petalWidth, 0, CSV_PETAL_WIDTH_IDX);
                }); var expectedOuputTensor = Tensor.of(TFloat32.class, Shape.of(1, OUTPUT_LAYER_WIDTH), data -> {
                    // 0 = 0%, 1 = 100% chance to be the expected species
                    // only 1 of the 3 must be set to 1, the rest 0
                    data.setFloat(currentInputData.irisSpecies == IrisSpecies.IRIS_SETOSA ? 1 : 0, 0,
                            OUTPUT_IRIS_SETOSA_IDX);
                    data.setFloat(currentInputData.irisSpecies == IrisSpecies.IRIS_VERSICOLOUR ? 1 : 0, 0,
                            OUTPUT_IRIS_VERSICOLOUR_IDX);
                    data.setFloat(currentInputData.irisSpecies == IrisSpecies.IRIS_VIRGINICA ? 1 : 0, 0,
                            OUTPUT_IRIS_VIRGINICA_IDX);
                })) {

                    Result result = tfSession.runner().addTarget(minimize)
                            .feed(OP_NAME_INPUT_LAYER_PLACEHOLDER, inputDataTensor)
                            .feed(trainingOutputPlaceholder, expectedOuputTensor).fetch(OP_NAME_OUTPUT_ACTIVATION)
                            .run();

                    var outputTensor = (TFloat32) result.get(0);
                    var chanceIrisSetosa = outputTensor.getFloat(0, OUTPUT_IRIS_SETOSA_IDX);
                    var chanceIrisVersicolour = outputTensor.getFloat(0, OUTPUT_IRIS_VERSICOLOUR_IDX);
                    var chanceIrisVirginica = outputTensor.getFloat(0, OUTPUT_IRIS_VIRGINICA_IDX);
                    var speciesToChanceMap = Map.of(IrisSpecies.IRIS_SETOSA, chanceIrisSetosa,
                            IrisSpecies.IRIS_VERSICOLOUR, chanceIrisVersicolour, IrisSpecies.IRIS_VIRGINICA,
                            chanceIrisVirginica);
                    var predictedSpecies = speciesToChanceMap.entrySet().stream().sorted((entry1, entry2) -> {
                        return entry1.getValue() > entry2.getValue() ? -1 : 1;
                    }).toList().get(0).getKey();
                    var predictedOk = predictedSpecies == currentInputData.irisSpecies;
                    if (predictedOk) {
                        numberOfPredictedOk++;
                    }

                    logger.info(" * Training epoch [{}], data index [{}]", currentTrainingEpoch, inputDataIdx);
                    logger.info("Training input: [{}]", currentInputData);
                    logger.info(
                            "Prediction: , predicted species [{}], chanceIrisSetosa [{}], chanceIrisVersicolour [{}], chanceIrisVirginica [{}]",
                            predictedSpecies, chanceIrisSetosa, chanceIrisVersicolour, chanceIrisVirginica);
                    logger.info("Predicted ok? [{}]", predictedOk);
                }
            }
            logger.info(" *** For training epoch [{}] predicted as expected for [{}]/[{}]", currentTrainingEpoch,
                    numberOfPredictedOk, trainData.size());
        }
    }

    public void buildNetwork(Ops tensorFlowApi) {
        var initializer = new Glorot<TFloat32>(Distribution.NORMAL, RANDOM_SEED);

        // input layer
        var inputLayerPlaceholder = tensorFlowApi.withName(OP_NAME_INPUT_LAYER_PLACEHOLDER).placeholder(TFloat32.class,
                Placeholder.shape(Shape.of(-1, INPUT_LAYER_WIDTH)));

        // hidden layer 1
        var hiddenLayer1Weights = tensorFlowApi.withName(OP_NAME_HIDDEN_LAYER1_WEIGHTS).variable(initializer
                .call(tensorFlowApi, tensorFlowApi.array(INPUT_LAYER_WIDTH, HIDDEN_LAYER_1_WIDTH), TFloat32.class));
        var hiddenLayer1Biases = tensorFlowApi.withName(OP_NAME_HIDDEN_LAYER1_BIASES)
                .variable(tensorFlowApi.fill(tensorFlowApi.array(HIDDEN_LAYER_1_WIDTH), tensorFlowApi.constant(0.1f)));
        var hiddenLayer1Activation = tensorFlowApi.nn.relu(tensorFlowApi.math
                .add(tensorFlowApi.linalg.matMul(inputLayerPlaceholder, hiddenLayer1Weights), hiddenLayer1Biases));

        // hidden layer 2
        var hiddenLayer2Weights = tensorFlowApi.withName(OP_NAME_HIDDEN_LAYER2_WEIGHTS).variable(initializer
                .call(tensorFlowApi, tensorFlowApi.array(HIDDEN_LAYER_1_WIDTH, HIDDEN_LAYER_2_WIDTH), TFloat32.class));
        var hiddenLayer2Biases = tensorFlowApi.withName(OP_NAME_HIDDEN_LAYER2_BIASES)
                .variable(tensorFlowApi.fill(tensorFlowApi.array(HIDDEN_LAYER_2_WIDTH), tensorFlowApi.constant(0.1f)));
        var hiddenLayer2Activation = tensorFlowApi.nn.relu(tensorFlowApi.math
                .add(tensorFlowApi.linalg.matMul(hiddenLayer1Activation, hiddenLayer2Weights), hiddenLayer2Biases));

        // output layer
        var outputLayerWeights = tensorFlowApi.withName(OP_NAME_OUTPUT_LAYER_WEIGHTS).variable(initializer
                .call(tensorFlowApi, tensorFlowApi.array(HIDDEN_LAYER_2_WIDTH, OUTPUT_LAYER_WIDTH), TFloat32.class));
        var outputLayerBiases = tensorFlowApi.withName(OP_NAME_OUTPUT_LAYER_BIASES)
                .variable(tensorFlowApi.fill(tensorFlowApi.array(OUTPUT_LAYER_WIDTH), tensorFlowApi.constant(0.1f)));
        tensorFlowApi.withName(OP_NAME_OUTPUT_ACTIVATION).nn.softmax(tensorFlowApi.math
                .add(tensorFlowApi.linalg.matMul(hiddenLayer2Activation, outputLayerWeights), outputLayerBiases));
    }

    private List<IrisDataLine> readTrainData() throws IOException {
        var trainData = new ArrayList<IrisDataLine>(INPUT_DATA_LINES);
        BufferedReader dataReader = new BufferedReader(
                new InputStreamReader(inputData.getInputStream(), StandardCharsets.UTF_8));
        while (dataReader.ready()) {
            var rawDataLine = dataReader.readLine();
            if (rawDataLine.isBlank()) {
                continue;
            }
            var splitDataLine = rawDataLine.split(CSV_SEPARATOR_REGEX);
            trainData.add(new IrisDataLine(Float.valueOf(splitDataLine[CSV_SEPAL_LENGTH_IDX]).floatValue(),
                    Float.valueOf(splitDataLine[CSV_SEPAL_WIDTH_IDX]).floatValue(),
                    Float.valueOf(splitDataLine[CSV_PETAL_LENGTH_IDX]).floatValue(),
                    Float.valueOf(splitDataLine[CSV_PETAL_WIDTH_IDX]).floatValue(),
                    IrisSpecies.getIrisSpecies(splitDataLine[CSV_SPECIES_NAME_IDX])));
        }

        return trainData;
    }
}
