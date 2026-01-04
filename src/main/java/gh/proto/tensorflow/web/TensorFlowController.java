package gh.proto.tensorflow.web;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import gh.proto.tensorflow.work.IrisClassifier;
import gh.proto.tensorflow.work.ObjectDetector;

@RestController
@RequestMapping("/tensorflow")
public class TensorFlowController {

    @Autowired
    private ObjectDetector objectDetector;

    @Autowired
    private IrisClassifier irisClassifier;

    @PostMapping(path = "/detect-objects", consumes = MediaType.APPLICATION_OCTET_STREAM_VALUE, produces = MediaType.APPLICATION_OCTET_STREAM_VALUE)
    public byte[] detectObjects(@RequestBody byte[] imageBytes) {

        return objectDetector.detect(imageBytes);
    }

    @GetMapping(path = "/iris-classify", produces = MediaType.TEXT_PLAIN_VALUE)
    public String irisClassify(@RequestParam float sepalLength, @RequestParam float sepalWidth,
            @RequestParam float petalLength, @RequestParam float petalWidth) {

        return irisClassifier.classify(sepalLength, sepalWidth, petalLength, petalWidth);
    }
}
