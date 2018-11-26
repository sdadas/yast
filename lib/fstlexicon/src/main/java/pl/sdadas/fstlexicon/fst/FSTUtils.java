package pl.sdadas.fstlexicon.fst;

import org.apache.commons.lang3.StringUtils;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.CharsRef;
import org.apache.lucene.util.IntsRef;
import org.apache.lucene.util.IntsRefBuilder;
import org.apache.lucene.util.fst.*;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Path;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.function.Function;

/**
 * Utilities for building finite state transducers (FSTs).
 *
 * @author Sławomir Dadas <sdadas@opi.org.pl>
 */
public class FSTUtils {

    /**
     * Creates new FST object from csv file.
     * example input file:
     *   word1,1
     *   word2,12
     *   word3,123
     * example usage:
     *   Function<String, FSTKeyValue> mapper = (line) -> FSTKeyValue.splitBy(line, ",");
     *   FST<Long> fst = </Long>csvToFst(new File("input.csv"), StandardChasets.UTF_8, mapper);
     *
     * @param inputFile file containing dictionary with corresponding numeric values
     * @param charset charset for input file
     * @param lineMapper a function that maps each line in the input file to FSTKeyValue object
     * @return an FST representation of input file
     */
    public static <T> FST<T> fromFile(File inputFile, Charset charset, Function<String, FSTKeyValue<T>> lineMapper) {
        SortedMap<String, T> map = new TreeMap<>();
        TextFiles.forEachLine(inputFile, charset, line -> {
            if(StringUtils.isBlank(line)) return;
            FSTKeyValue<T> kv = lineMapper.apply(line);
            map.put(kv.key(), kv.value());
        });
        return build(map);
    }

    /**
     * Creates new FST object from file.
     */
    public static <T> FST<T> fromFile(String inputPath, Charset charset, Function<String, FSTKeyValue<T>> lineMapper) {
        return fromFile(new File(inputPath), charset, lineMapper);
    }

    /**
     * Creates new FST object from file.
     */
    public static <T> FST<T> fromFile(Path inputFile, Charset charset, Function<String, FSTKeyValue<T>> lineMapper) {
        return fromFile(inputFile.toFile(), charset, lineMapper);
    }

    /**
     * Creates new FST object from SortedMap.
     *
     * @param map map with words and corresponding numeric values
     * @return an FST representation of map
     */
    public static <T> FST<T> build(SortedMap<String, T> map) {
        try {
            return createNewFst(map);
        } catch (IOException e) {
            throw new IllegalStateException("Error building FST dictionary", e);
        }
    }

    /**
     * Loads FST object from file.
     *
     * @param fstFile FST file
     * @return loaded FST object
     */
    public static <T> FST<T> load(File fstFile, Class<T> clazz) {
        return load(fstFile.toPath(), clazz);
    }

    /**
     * Loads FST object from file.
     *
     * @param fstFile FST file
     * @return loaded FST object
     */
    public static <T> FST<T> load(Path fstFile, Class<T> clazz) {
        try {
            return FST.read(fstFile, createOutputs(clazz));
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    private static <T> FST<T> createNewFst(SortedMap<String, T> map) throws IOException {
        T firstValue = map.get(map.firstKey());
        Builder<T> builder = new Builder<>(FST.INPUT_TYPE.BYTE4, createOutputs(firstValue.getClass()));
        for (Map.Entry<String, T> entry : map.entrySet()) {
            IntsRef word = Util.toIntsRef(new BytesRef(entry.getKey()), new IntsRefBuilder());
            builder.add(word, entry.getValue());
        }
        return builder.finish();
    }

    @SuppressWarnings("unchecked")
    private static  <T> Outputs<T> createOutputs(Class<?> clazz) {
        if(clazz.equals(Long.class)) {
            return (Outputs<T>) PositiveIntOutputs.getSingleton();
        } else if(clazz.equals(CharsRef.class)) {
            return (Outputs<T>) CharSequenceOutputs.getSingleton();
        } else {
            throw new IllegalStateException("Unsupported output class " + clazz.getSimpleName());
        }
    }

    public static <T> T get(FST<T> fst, String input) {
        try {
            return get(fst, new BytesRef(input));
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    public static <T> boolean hasPrefix(FST<T> fst, String prefix) {
        try {
            return hasPrefix(fst, new BytesRef(prefix));
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    public static <T> boolean hasPrefix(FST<T> fst, BytesRef input) throws IOException {
        final FST.BytesReader fstReader = fst.getBytesReader();
        final FST.Arc<T> arc = fst.getFirstArc(new FST.Arc<T>());
        for(int i=0;i<input.length;i++) {
            if (fst.findTargetArc(input.bytes[i+input.offset] & 0xFF, arc, arc, fstReader) == null) {
                return false;
            }
        }
        return true;
    }

    private static <T> T get(FST<T> fst, BytesRef input) throws IOException {
        final FST.BytesReader fstReader = fst.getBytesReader();
        final FST.Arc<T> arc = fst.getFirstArc(new FST.Arc<T>());
        T output = fst.outputs.getNoOutput();
        for(int i=0;i<input.length;i++) {
            if (fst.findTargetArc(input.bytes[i+input.offset] & 0xFF, arc, arc, fstReader) == null) {
                return null;
            }
            output = fst.outputs.add(output, arc.output);
        }
        if (arc.isFinal()) {
            return fst.outputs.add(output, arc.nextFinalOutput);
        } else {
            return null;
        }
    }
}
