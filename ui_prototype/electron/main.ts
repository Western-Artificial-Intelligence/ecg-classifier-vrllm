import { app, BrowserWindow, shell } from "electron";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const devServerUrl = process.env.VITE_DEV_SERVER_URL ?? "http://localhost:5173";
const isDev = !app.isPackaged;

function createWindow() {
  const win = new BrowserWindow({
    width: 1320,
    height: 820,
    minWidth: 1120,
    minHeight: 720,
    backgroundColor: "#05070c",
    title: "SomniScope",
    show: false,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  });

  win.webContents.on("did-fail-load", (_event, errorCode, errorDescription, validatedURL) => {
    // This is the #1 reason you get a black screen in Electron dev.
    console.error("[did-fail-load]", { errorCode, errorDescription, validatedURL });
  });

  win.webContents.on("render-process-gone", (_event, details) => {
    console.error("[render-process-gone]", details);
  });

  win.webContents.on("console-message", (_event, level, message, line, sourceId) => {
    // Helpful to surface renderer crashes to the terminal.
    console.log("[renderer]", { level, message, line, sourceId });
  });

  win.once("ready-to-show", () => win.show());

  win.webContents.setWindowOpenHandler(({ url }) => {
    void shell.openExternal(url);
    return { action: "deny" };
  });

  if (isDev) {
    console.log("[dev] loading", devServerUrl);
    win.loadURL(devServerUrl).catch((err) => console.error("[loadURL error]", err));
    win.webContents.openDevTools({ mode: "detach" });
  } else {
    win
      .loadFile(path.join(process.cwd(), "dist", "index.html"))
      .catch((err) => console.error("[loadFile error]", err));
  }
}

app.whenReady().then(() => {
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});


