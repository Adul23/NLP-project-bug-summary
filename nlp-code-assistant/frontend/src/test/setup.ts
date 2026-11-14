import "@testing-library/jest-dom/vitest";
import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

afterEach(() => cleanup());

// jsdom has no layout; CodeMirror measures text ranges, so give it empty geometry.
const emptyRect = () => ({ x: 0, y: 0, top: 0, left: 0, bottom: 0, right: 0, width: 0, height: 0, toJSON() {} });
Range.prototype.getBoundingClientRect = emptyRect as unknown as Range["getBoundingClientRect"];
Range.prototype.getClientRects = (() => {
  const list = [] as unknown as DOMRectList;
  return list;
}) as Range["getClientRects"];
