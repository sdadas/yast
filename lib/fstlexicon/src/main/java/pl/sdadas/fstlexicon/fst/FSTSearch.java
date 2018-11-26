package pl.sdadas.fstlexicon.fst;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.mutable.MutableInt;
import org.apache.lucene.util.fst.FST;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class FSTSearch<T> {

    private final FST<T> fst;

    private Set<String> stopwords = new HashSet<>();

    public FSTSearch(FST<T> fst) {
        this.fst = fst;
    }

    public FSTSearch(FST<T> fst, Set<String> stopwords) {
        this.fst = fst;
        this.stopwords = stopwords;
    }

    public T matches(String value) {
         return FSTUtils.get(fst, value);
    }

    public List<FSTMatch<T>> find(String[] words) {
        List<FSTMatch<T>> matches = new ArrayList<>();
        MutableInt idx = new MutableInt(0);
        while(idx.intValue() < words.length) {
            FSTMatch<T> match = findNextMatch(idx, words);
            if(match != null) matches.add(match);
        }
        return matches;
    }

    private FSTMatch<T> findNextMatch(MutableInt startIdx, String[] words) {
        int idx;
        for (idx = startIdx.intValue(); idx < words.length; idx++) {
            String word = words[idx];
            T result = FSTUtils.get(this.fst, word);
            if(result != null && !stopwords.contains(word)) {
                startIdx.setValue(idx + 1);
                return new FSTMatch<T>(idx, idx, result);
            } else if(FSTUtils.hasPrefix(this.fst, word)) {
                FSTMatch<T> match = findMultiWordMatch(idx, words);
                if(match != null) {
                    startIdx.setValue(match.endIdx + 1);
                    return match;
                }
            }
        }
        startIdx.setValue(idx);
        return null;
    }

    private FSTMatch<T> findMultiWordMatch(int startIdx, String[] words) {
        for (int i = 1; i < 20; i++) {
            FSTMatch<T> res = findMultiWordMatch(startIdx, startIdx + i, words);
            if(res.value != null) return res;
            else if(!res.prefix) return null;
        }
        return null;
    }

    private FSTMatch<T> findMultiWordMatch(int startIdx, int endIdx, String[] words) {
        FSTMatch<T> result = new FSTMatch<>(startIdx, endIdx);
        if(endIdx >= words.length) return result;

        String source = getMultiWordValue(startIdx, endIdx, words);
        T target = FSTUtils.get(this.fst, source);
        if(target != null && !stopwords.contains(source)) {
            result.value = target;
            return result;
        } else if(FSTUtils.hasPrefix(this.fst, source)) {
            result.prefix = true;
            return result;
        }
        return result;
    }

    private String getMultiWordValue(int startIdx, int endIdx, String[] words) {
        String value = String.join(" ", ArrayUtils.subarray(words, startIdx, endIdx + 1));
        return value.replace(" - ", "-").replace(" ,", ",");
    }
}
