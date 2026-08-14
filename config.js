/**
 * Deployment configuration
 *
 * Backend is hosted on Azure App Service (Basic B1, Azure for Students).
 * Local dev (python server.py on port 5001) works without changing anything.
 */

// // Local dev
// window.TRACKING_API  = 'http://localhost:5001';
// window.IMAGES_BASE   = 'http://localhost:5001';

// Azure App Service backend URL
window.TRACKING_API  = 'https://spotdafake-b6f0c7geecambmhv.centralus-01.azurewebsites.net/';
window.IMAGES_BASE   = 'https://spotdafake-b6f0c7geecambmhv.centralus-01.azurewebsites.net/';
