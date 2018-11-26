import org.apache.lucene.util.fst.FST;
import pl.sdadas.fstlexicon.fst.FSTUtils;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.util.SortedMap;
import java.util.TreeMap;

/**
 * @author Sławomir Dadas
 */
public class CreateTestData {

    public static void main(String [] args) throws IOException {
        FST<Long> fst = FSTUtils.build(createMap());
        fst.save(new File("food.fst").toPath());
        byte[] meta = createMetaJson().getBytes(StandardCharsets.UTF_8);
        Files.write(new File("meta.json").toPath(), meta, StandardOpenOption.CREATE);
    }

    private static SortedMap<String, Long> createMap() {
        SortedMap<String, Long> res = new TreeMap<>();
        res.put("pizza", 0L);
        res.put("gnocchi", 0L);
        res.put("pasta", 0L);
        res.put("pierogi", 1L);
        res.put("bigos", 1L);
        res.put("kiełbasa", 1L);
        res.put("paella", 2L);
        res.put("gazpacho", 2L);
        res.put("chorizo", 2L);
        res.put("sushi", 3L);
        res.put("ramen", 3L);
        res.put("miso", 3L);
        return res;
    }

    private static String createMetaJson() {
        return "{\"food\":[\"italian\", \"polish\", \"spanish\", \"japanese\", \"other\"]}";
    }
}
