package gh.proto.tensorflow.work;

import java.util.Map;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.tensorflow.Result;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.types.TFloat32;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;

@Component
public class IrisClassifier {

    private final Logger logger = LogManager.getLogger(getClass());

    // must match the order used during training
    private static final int SEPAL_LENGTH_IDX = 0;
    private static final int SEPAL_WIDTH_IDX = 1;
    private static final int PETAL_LENGTH_IDX = 2;
    private static final int PETAL_WIDTH_IDX = 3;

    // chosen by us and must be consistent thru training and running
    private static final String IRIS_SETOSA = "Iris-setosa";
    private static final int OUTPUT_IRIS_SETOSA_IDX = 0;

    private static final String IRIS_VERSICOLOUR = "Iris-versicolor";
    private static final int OUTPUT_IRIS_VERSICOLOUR_IDX = 1;

    private static final String IRIS_VIRGINICA = "Iris-virginica";
    private static final int OUTPUT_IRIS_VIRGINICA_IDX = 2;

    private static final long INPUT_LAYER_WIDTH = 4L;
    private static final String OP_NAME_INPUT_LAYER_PLACEHOLDER = "inputLayerPlaceholder";
    private static final String OP_NAME_OUTPUT_ACTIVATION = "outputActivation";

    @Value("${tensorflow.iris.path}")
    private String modelPath;

    private SavedModelBundle model;

    @PostConstruct
    public void init() {
        model = SavedModelBundle.load(modelPath, SavedModelBundle.DEFAULT_TAG);

        logger.info("TensorFlow model functions: [{}]", model.signatures());
    }

    @PreDestroy
    public void clean() {
        model.close();
    }

    public String classify(float sepalLength, float sepalWidth, float petalLength, float petalWidth) {
        String species = null;
        Session tfSession = model.session();
        try (var inputDataTensor = Tensor.of(TFloat32.class, Shape.of(1, INPUT_LAYER_WIDTH), data -> {
            data.setFloat(sepalLength, 0, SEPAL_LENGTH_IDX);
            data.setFloat(sepalWidth, 0, SEPAL_WIDTH_IDX);
            data.setFloat(petalLength, 0, PETAL_LENGTH_IDX);
            data.setFloat(petalWidth, 0, PETAL_WIDTH_IDX);

        })) {
            Result result = tfSession.runner().feed(OP_NAME_INPUT_LAYER_PLACEHOLDER, inputDataTensor)
                    .fetch(OP_NAME_OUTPUT_ACTIVATION).run();

            var outputTensor = (TFloat32) result.get(0);
            var chanceIrisSetosa = outputTensor.getFloat(0, OUTPUT_IRIS_SETOSA_IDX);
            var chanceIrisVersicolour = outputTensor.getFloat(0, OUTPUT_IRIS_VERSICOLOUR_IDX);
            var chanceIrisVirginica = outputTensor.getFloat(0, OUTPUT_IRIS_VIRGINICA_IDX);
            var speciesToChanceMap = Map.of(IRIS_SETOSA, chanceIrisSetosa, IRIS_VERSICOLOUR, chanceIrisVersicolour,
                    IRIS_VIRGINICA, chanceIrisVirginica);
            species = speciesToChanceMap.entrySet().stream().sorted((entry1, entry2) -> {
                return entry1.getValue() > entry2.getValue() ? -1 : 1;
            }).toList().get(0).getKey();

            logger.info(
                    " For sepalLength [{}], sepalWidth [{}], petalLength [{}], petalWidth [{}] the predicted species is [{}]",
                    sepalLength, sepalWidth, petalLength, petalWidth, species);
        }

        return species;
    }
}
