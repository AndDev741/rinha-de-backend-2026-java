package com.andre.rinha.http;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

import java.io.IOException;

/**
 * GET /ready — health check.
 *
 * Responds 200 with empty body when the app is ready. "Ready" means the
 * dataset is loaded — controlled by the initialization order in App.java
 * (we register the handler only AFTER load completes).
 */
public final class ReadyHandler implements HttpHandler {
    private static final byte[] EMPTY = new byte[0];

    @Override
    public void handle(HttpExchange exchange) throws IOException {
        exchange.sendResponseHeaders(200, -1); // -1 = no body
        exchange.close();
    }
}
