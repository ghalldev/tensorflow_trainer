package gh.proto.tensorflow.work;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.tensorflow.Graph;
import org.tensorflow.Result;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Reshape;
import org.tensorflow.op.image.DecodeImage;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TString;
import org.tensorflow.types.TUint8;

import jakarta.annotation.PostConstruct;

@Component
public class ObjectDetector {

    private final Logger logger = LogManager.getLogger(getClass());

    private record ObjectBox(int idx, float ymin, float xmin, float ymax, float xmax) {
    };

    private static final float DETECTION_SENSITIVITY = 0.3f;

    private final static String[] COCO_LABELS = new String[] { "index_shifter_value", "person", "bicycle", "car",
            "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "street sign",
            "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
            "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase",
            "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet", "door", "tv",
            "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
            "hair brush" };

    @Value("${tensorflow.objectdetection.path}")
    private String modelPath;

    private SavedModelBundle model;

    @PostConstruct
    public void init() {
        model = SavedModelBundle.load(modelPath, SavedModelBundle.DEFAULT_TAG);

        logger.info("TensorFlow model functions: [{}]", model.signatures());
    }

    public byte[] detect(byte[] imageData) {
        var response = runModel(imageData);

        logger.info("Detection data: [{}]", response);

        return response.toString().getBytes();
    }

    private List<ObjectBox> runModel(byte[] imageData) {
        var objectBoxes = new ArrayList<ObjectBox>();

        try (Graph tfGtaph = new Graph(); Session tfSession = new Session(tfGtaph)) {
            Ops tensorFlowApi = Ops.create(tfGtaph);
            Session.Runner runner = tfSession.runner();
            DecodeImage.Options[] options = { DecodeImage.channels(3L) };
            // https://discuss.ai.google.dev/t/decode-jpeg-from-byte/31508
            DecodeImage<TUint8> decodeImage = tensorFlowApi.image.decodeImage(
                    tensorFlowApi.constant(TString.tensorOfBytes(NdArrays.scalarOfObject(imageData))), options);
            Shape imageShape;
            try (var shapeResult = runner.fetch(decodeImage).run()) {
                imageShape = shapeResult.get(0).shape();
            }
            Reshape<TUint8> reshape = tensorFlowApi.reshape(decodeImage,
                    tensorFlowApi.array(1, imageShape.asArray()[0], imageShape.asArray()[1], 3));
            try (var reshapeResult = tfSession.runner().fetch(reshape).run()) {
                TUint8 reshapeTensor = (TUint8) reshapeResult.get(0);
                Map<String, Tensor> feedDict = new HashMap<>();
                feedDict.put("input_tensor", reshapeTensor);
                try (Result outputTensorMap = model.function("serving_default").call(feedDict)) {
                    TFloat32 numDetections = (TFloat32) outputTensorMap.get("num_detections").get();
                    int numDetects = (int) numDetections.getFloat(0);
                    if (numDetects > 0) {
                        TFloat32 detectionScores = (TFloat32) outputTensorMap.get("detection_scores").get();
                        TFloat32 detectionBoxes = (TFloat32) outputTensorMap.get("detection_boxes").get();
                        TFloat32 detectionClasses = (TFloat32) outputTensorMap.get("detection_classes").get();
                        for (int n = 0; n < numDetects; n++) {
                            float detectionScore = detectionScores.getFloat(0, n);
                            if (detectionScore > DETECTION_SENSITIVITY) {
                                var modelBox = detectionBoxes.get(0, n);
                                var objectBox = new ObjectBox((int) detectionClasses.getFloat(0, n),
                                        modelBox.getFloat(0), modelBox.getFloat(1), modelBox.getFloat(2),
                                        modelBox.getFloat(3));
                                logger.info("Current box: [{}], label: [{}]", objectBox, COCO_LABELS[objectBox.idx]);
                                objectBoxes.add(objectBox);
                            }
                        }
                    }
                }
            }
        }
        return objectBoxes;
    }
}
