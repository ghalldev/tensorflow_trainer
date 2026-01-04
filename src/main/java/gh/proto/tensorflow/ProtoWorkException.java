package gh.proto.tensorflow;

public class ProtoWorkException extends RuntimeException {

    public ProtoWorkException(String message) {
        super(message);
    }

    public ProtoWorkException(String message, Throwable e) {
        super(message, e);
    }
}
