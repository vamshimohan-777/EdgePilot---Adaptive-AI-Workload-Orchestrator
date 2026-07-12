package edgepilot;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

// =============================================================================
// EdgePilot — P4: Platform Layer
// EdgePilotClient.java — Java Client SDK binding for Java applications.
// =============================================================================

public class EdgePilotClient {
    private final String apiUrl;
    private final HttpClient httpClient;

    public EdgePilotClient(String apiUrl) {
        this.apiUrl = apiUrl.replaceAll("/$", "");
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(2))
                .build();
    }

    public EdgePilotClient() {
        this("http://127.0.0.1:8000");
    }

    /**
     * Submits a job to the scheduling queue.
     *
     * @param modelId  Target model ID
     * @param prompt   Text prompt/input
     * @param priority Priority level (0=Low, 1=Normal, 2=High, 3=RealTime)
     * @return The generated Job ID string
     */
    public String submitJob(String modelId, String prompt, int priority) throws IOException, InterruptedException {
        String jsonPayload = String.format(
                "{\"model_id\":\"%s\",\"priority\":%d,\"prompt\":\"%s\"}",
                modelId, priority, prompt.replace("\"", "\\\"")
        );

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(apiUrl + "/jobs/submit"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonPayload))
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() != 200) {
            throw new IOException("Server returned error status: " + response.statusCode() + " - " + response.body());
        }

        // Simple JSON parsing to extract "job_id"
        String body = response.body();
        int keyIndex = body.indexOf("\"job_id\"");
        if (keyIndex == -1) {
            throw new IOException("Malformed response: " + body);
        }
        int colonIndex = body.indexOf(":", keyIndex);
        int startIndex = body.indexOf("\"", colonIndex);
        int endIndex = body.indexOf("\"", startIndex + 1);
        return body.substring(startIndex + 1, endIndex);
    }

    /**
     * Queries the status of a job.
     *
     * @param jobId The job identifier
     * @return Raw JSON response containing job status
     */
    public String queryStatus(String jobId) throws IOException, InterruptedException {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(apiUrl + "/jobs/status/" + jobId))
                .GET()
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        return response.body();
    }

    /**
     * Synchronously runs an inference job, blocking until completion or timeout.
     */
    public String runInference(String modelId, String prompt, int priority, long timeoutMs) 
            throws IOException, InterruptedException, TimeoutException {
        String jobId = submitJob(modelId, prompt, priority);
        
        long startTime = System.currentTimeMillis();
        while (System.currentTimeMillis() - startTime < timeoutMs) {
            String statusJson = queryStatus(jobId);
            
            if (statusJson.contains("\"COMPLETED\"")) {
                int keyIndex = statusJson.indexOf("\"result\"");
                int colonIndex = statusJson.indexOf(":", keyIndex);
                int startIndex = statusJson.indexOf("\"", colonIndex);
                int endIndex = statusJson.indexOf("\"", startIndex + 1);
                return statusJson.substring(startIndex + 1, endIndex).replace("\\\"", "\"");
            } else if (statusJson.contains("\"FAILED\"")) {
                throw new RuntimeException("EdgePilot inference job '" + jobId + "' failed.");
            }
            
            Thread.sleep(200);
        }
        
        throw new TimeoutException("Job '" + jobId + "' timed out after " + timeoutMs + " ms.");
    }

    public static class TimeoutException extends Exception {
        public TimeoutException(String message) {
            super(message);
        }
    }
}
