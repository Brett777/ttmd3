import React, { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { ChakraProvider } from "@chakra-ui/react";
import { DocumentProvider } from "./contexts/DocumentContext";
import { DataProvider } from "./contexts/DataContext";

import "./index.css";
import App from "./App";

// Add this line to handle BigInt serialization
BigInt.prototype.toJSON = function() { return this.toString(); };

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <ChakraProvider>
      <DocumentProvider>
        <DataProvider>
          <App />
        </DataProvider>
      </DocumentProvider>
    </ChakraProvider>
  </StrictMode>
);
