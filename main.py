import os
import threading
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.utils import platform

# Import your Flask app
# This path is relative to the new main.py file
from backend.app import app as flask_app

# --- Kivy UI and Flask Server Logic ---
class MainApp(App):
    def build(self):
        # Start the Flask server in a new thread
        threading.Thread(target=self.start_flask_server, daemon=True).start()

        # The Kivy UI will be a simple web view
        if platform == 'android':
            return self.create_android_ui()
        else:
            return Label(text='Running on non-Android platform')

    def start_flask_server(self):
        # Run the Flask app
        flask_app.run(host='0.0.0.0', port=5000)

    def create_android_ui(self):
        # This is a placeholder for a Kivy WebView
        # It will be a more complex setup using Java classes
        # Refer to the previous detailed instructions on how to set this up.
        from jnius import autoclass
        from android.runnable import run_on_ui_thread

        @run_on_ui_thread
        def create_webview_on_ui():
            WebView = autoclass('android.webkit.WebView')
            WebViewClient = autoclass('android.webkit.WebViewClient')
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            activity = PythonActivity.mActivity

            webview = WebView(activity)
            webview.getSettings().setJavaScriptEnabled(True)
            webview.setWebViewClient(WebViewClient())
            webview.loadUrl('http://127.0.0.1:5000')
            activity.addContentView(webview, Window.width, Window.height)

        create_webview_on_ui()
        return BoxLayout() # Return a blank layout for Kivy's main window

if __name__ == '__main__':
    MainApp().run()