package pl.sdadas.fstlexicon.fst;

/**
 * @author Sławomir Dadas <sdadas@opi.org.pl>
 */
public class FSTKeyValue<T> {

    private final String key;

    private final T value;

    public FSTKeyValue(String key, T value) {
        this.key = key;
        this.value = value;
    }

    public String key() {
        return key;
    }

    public T value() {
        return value;
    }
}
