#pragma once

#include <string>
#include <sstream>
#include <iostream>
#include <chrono>
#include <thread>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

// =============================================================================
// EdgePilot — P4: Platform Layer
// edgepilot_client.h — C++ Native Client SDK binding.
//
// Directly communicates with the EdgePilot C++ daemon over TCP socket loopback
// for high-speed, zero-dependency SDK execution.
// =============================================================================

namespace edgepilot {

class EdgePilotClient {
public:
    explicit EdgePilotClient(const std::string& host = "127.0.0.1", int port = 12345)
        : host_(host)
        , port_(port)
    {
#ifdef _WIN32
        WSADATA wsaData;
        int err = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (err != 0) {
            throw std::runtime_error("WSAStartup failed: " + std::to_string(err));
        }
#endif
    }

    ~EdgePilotClient() {
#ifdef _WIN32
        WSACleanup();
#endif
    }

    /// Submits a job to the scheduling queue.
    std::string SubmitJob(const std::string& model_id, const std::string& prompt, int priority = 1) {
        std::string job_id = "cpp-" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::system_clock::now().time_since_epoch()).count() % 1000000);
        
        std::stringstream ss;
        ss << "SUBMIT " << job_id << " " << model_id << " " << priority << " " << prompt;
        std::string reply = SendCommand(ss.str());

        if (reply.rfind("OK", 0) == 0) {
            return job_id;
        }
        throw std::runtime_error("Daemon rejected submission: " + reply);
    }

    /// Queries the current status of a job.
    std::string QueryStatus(const std::string& job_id) {
        return SendCommand("STATUS " + job_id);
    }

    /// Synchronously runs an inference job, blocking until completion or timeout.
    std::string RunInference(const std::string& model_id, const std::string& prompt, int priority = 1, int timeout_ms = 10000) {
        std::string job_id = SubmitJob(model_id, prompt, priority);

        auto start = std::chrono::steady_clock::now();
        while (true) {
            std::string status = QueryStatus(job_id);

            if (status.rfind("COMPLETED", 0) == 0) {
                // Return completion payload: strip "COMPLETED " prefix
                if (status.length() > 10) {
                    return status.substr(10);
                }
                return "Success";
            }
            if (status.rfind("FAILED", 0) == 0) {
                throw std::runtime_error("Inference job failed inside EdgePilot daemon.");
            }

            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
            if (elapsed > timeout_ms) {
                throw std::runtime_error("Inference job timed out after " + std::to_string(timeout_ms) + " ms.");
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }

private:
    std::string SendCommand(const std::string& cmd) {
#ifdef _WIN32
        SOCKET s = socket(AF_INET, SOCK_STREAM, 0);
        if (s == INVALID_SOCKET) {
            throw std::runtime_error("Failed to create socket");
        }
#else
        int s = socket(AF_INET, SOCK_STREAM, 0);
        if (s < 0) {
            throw std::runtime_error("Failed to create socket");
        }
#endif

        sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port_);
        inet_pton(AF_INET, host_.c_str(), &addr.sin_addr);

#ifdef _WIN32
        if (connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
            closesocket(s);
            throw std::runtime_error("Could not connect to EdgePilot daemon.");
        }
#else
        if (connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            ::close(s);
            throw std::runtime_error("Could not connect to EdgePilot daemon.");
        }
#endif

        send(s, cmd.c_str(), static_cast<int>(cmd.length()), 0);

        char buffer[4096] = {0};
        int bytes = recv(s, buffer, sizeof(buffer) - 1, 0);
        
        std::string reply = "";
        if (bytes > 0) {
            reply = std::string(buffer, bytes);
            // strip trailing newlines
            while (!reply.empty() && (reply.back() == '\n' || reply.back() == '\r')) {
                reply.pop_back();
            }
        }

#ifdef _WIN32
        closesocket(s);
#else
        ::close(s);
#endif
        return reply;
    }

    std::string host_;
    int         port_;
};

} // namespace edgepilot
