import { contextBridge } from "electron";
contextBridge.exposeInMainWorld("somni", {
    platform: process.platform
});
//# sourceMappingURL=preload.js.map