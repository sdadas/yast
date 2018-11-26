package pl.sdadas.fstlexicon.fst;

public class FSTMatch<T> {

    final int startIdx;

    final int endIdx;

    T value;

    boolean prefix;

    FSTMatch(int startIdx, int endIdx) {
        this.startIdx = startIdx;
        this.endIdx = endIdx;
    }

    FSTMatch(int startIdx, int endIdx, T value) {
        this.startIdx = startIdx;
        this.endIdx = endIdx;
        this.value = value;
    }

    public int getStartIdx() {
        return startIdx;
    }

    public int getEndIdx() {
        return endIdx;
    }

    public T getValue() {
        return value;
    }
}
