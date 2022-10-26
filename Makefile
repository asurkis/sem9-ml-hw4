default: clean task.py.zip

clean:
	rm -rf *.zip

task.py.zip: task.py
	zip $@ $+
