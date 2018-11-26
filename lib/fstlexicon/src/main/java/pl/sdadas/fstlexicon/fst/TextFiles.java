package pl.sdadas.fstlexicon.fst;

import org.apache.commons.lang3.StringUtils;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.function.Consumer;
import java.util.stream.Stream;

/**
 * @author Sławomir Dadas <sdadas@opi.org.pl>
 */
public final class TextFiles {

    public static void forEachLine(String inputPath, Charset charset, Consumer<String> consumer) {
        forEachLine(new File(inputPath), charset, consumer);
    }

    public static void forEachLine(File inputFile, Charset charset, Consumer<String> consumer) {
        try(Stream<String> lines = Files.lines(inputFile.toPath(), charset)) {
            lines.forEach(consumer);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    public static void forEachRecord(String inputPath, Charset charset, String splitBy, Consumer<String[]> consumer) {
        forEachRecord(new File(inputPath), charset, splitBy, consumer);
    }

    public static void forEachRecord(File inputFile, Charset charset, String splitBy, Consumer<String[]> consumer) {
        Consumer<String> lineConsumer = (line) -> consumer.accept(StringUtils.split(line, splitBy));
        forEachLine(inputFile, charset, lineConsumer);
    }

    public static void writeLines(String outputPath, Charset charset, Stream<String> stream) {
        writeLines(new File(outputPath), charset, stream);
    }

    public static void writeLines(File outputFile, Charset charset, Stream<String> stream) {
        try(PrintWriter writer = new PrintWriter(outputFile, charset.name())) {
            stream.forEach(writer::println);
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            throw new IllegalStateException(e);
        }
    }

    private TextFiles() {
    }
}
