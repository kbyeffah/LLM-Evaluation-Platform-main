// pages/api/proxy/[...path].js
import httpProxyMiddleware from 'http-proxy-middleware';

export const config = {
  api: {
    bodyParser: false,
    externalResolver: true,
  },
};

const proxy = httpProxyMiddleware.createProxyMiddleware({
  target: 'http://localhost:5000',
  changeOrigin: true,
  pathRewrite: {
    '^/api/proxy': '', // Remove the /api/proxy prefix when forwarding to backend
  },
});

export default function handler(req, res) {
  return proxy(req, res);
}