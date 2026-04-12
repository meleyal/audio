APPS = drums midi patch slice stems voice

$(APPS):
	cd $@ && uv run --with-requirements requirements.txt gradio app.py

.PHONY: $(APPS)
